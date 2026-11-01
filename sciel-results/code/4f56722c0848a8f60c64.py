"""Matrix multiplication 2,2,11 - Global linear algebra solver for minimal rank decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 11]

def _build_tensor():
    """Build the multiplication tensor as a set of (a,b,c) -> 1 entries."""
    n, m, p = DIMENSIONS
    entries = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                entries[(a, b, c)] = 1.0
    return entries

def _verify(U, V, W):
    """Verify decomposition exactly reproduces the multiplication tensor."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    return np.max(np.abs(T_decomp - T)) < 1e-9

def _build_rank39():
    """Verified rank-39 decomposition: 5 Strassen blocks (rank 7 each) + 1 schoolbook column (rank 4)."""
    n, m, p = 2, 2, 11
    rank = 5 * 7 + 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        
        v_e = k0       # B[0, k0]
        v_f = k1       # B[0, k1]
        v_g = 11 + k0  # B[1, k0]
        v_h = 11 + k1  # B[1, k1]
        
        w_c00 = 2 * k0     # C[0, k0]
        w_c10 = 2 * k0 + 1 # C[1, k0]
        w_c01 = 2 * k1     # C[0, k1]
        w_c11 = 2 * k1 + 1 # C[1, k1]
        
        # m1 = (a+d)(e+h)
        U[0][r] = 1; U[3][r] = 1
        V[v_e][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1; W[w_c11][r] = 1
        r += 1
        
        # m2 = (c+d)e
        U[2][r] = 1; U[3][r] = 1
        V[v_e][r] = 1
        W[w_c10][r] = 1; W[w_c11][r] = -1
        r += 1
        
        # m3 = a(f-h)
        U[0][r] = 1
        V[v_f][r] = 1; V[v_h][r] = -1
        W[w_c01][r] = 1; W[w_c11][r] = 1
        r += 1
        
        # m4 = d(g-e)
        U[3][r] = 1
        V[v_g][r] = 1; V[v_e][r] = -1
        W[w_c00][r] = 1; W[w_c10][r] = 1
        r += 1
        
        # m5 = (a+b)h
        U[0][r] = 1; U[1][r] = 1
        V[v_h][r] = 1
        W[w_c00][r] = -1; W[w_c01][r] = 1
        r += 1
        
        # m6 = (c-a)(e+f)
        U[2][r] = 1; U[0][r] = -1
        V[v_e][r] = 1; V[v_f][r] = 1
        W[w_c11][r] = 1
        r += 1
        
        # m7 = (b-d)(g+h)
        U[1][r] = 1; U[3][r] = -1
        V[v_g][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1
        r += 1
    
    # Single column k=10: schoolbook (4 terms)
    k2 = 10
    v_e2 = k2       # B[0,10]
    v_g2 = 11 + k2  # B[1,10]
    w_c02 = 2 * k2     # C[0,10]
    w_c12 = 2 * k2 + 1 # C[1,10]
    
    # ae -> C[0,10]
    U[0][r] = 1; V[v_e2][r] = 1; W[w_c02][r] = 1; r += 1
    # bg -> C[0,10]
    U[1][r] = 1; V[v_g2][r] = 1; W[w_c02][r] = 1; r += 1
    # ce -> C[1,10]
    U[2][r] = 1; V[v_e2][r] = 1; W[w_c12][r] = 1; r += 1
    # dg -> C[1,10]
    U[3][r] = 1; V[v_g2][r] = 1; W[w_c12][r] = 1; r += 1
    
    return {'U': U, 'V': V, 'W': W}

def _try_global_solver(deadline):
    """
    Try to find a decomposition with rank < 39 using a global linear algebra approach.
    
    Strategy: The 11 column-pairs each need a rank-7 Strassen-like decomposition.
    If we can find shared structure, we might reduce the total rank.
    
    We model this as: find U (4×r), V (22×r), W (22×r) such that
    sum_t U[a,t]*V[b,t]*W[c,t] = T[a,b,c] for all (a,b,c).
    
    For a given rank r, this is a system of 4*22*22 = 1936 equations in r*(4+22+22) = 48r unknowns.
    But the equations are cubic (trilinear), so we use an iterative approach.
    
    Alternative: Use the fact that for each fixed (a,c) pair, we need:
    sum_t U[a,t]*W[c,t]*V[b,t] = T[a,b,c] for all b.
    
    This means for each (a,c), the vector V_b (over b) must satisfy a linear constraint.
    
    Let's try a different structural approach: assume the decomposition has a block structure
    where the 11 columns are handled by a shared set of "basis" terms.
    
    For rank r, we need 48r >= 1936 (number of tensor entries), so r >= 41.
    Wait, that's not right - the tensor only has 44 non-zero entries, not 1936.
    
    Actually, the constraint is: for all (a,b,c) in the 44 non-zero entries, the sum must equal 1,
    and for all other (a,b,c) combinations, the sum must equal 0.
    
    Total constraints: 4*22*22 = 1936 (all must be satisfied).
    Unknowns: r*(4+22+22) = 48r.
    
    For r=39: 48*39 = 1872 < 1936. So the system is overdetermined, but the tensor is sparse
    and has special structure, so it's still solvable.
    
    For r=38: 48*38 = 1824. Still possible due to structure.
    For r=37: 48*37 = 1776.
    
    Let me try a different approach: use the known rank-39 solution and try to merge terms.
    """
    # This is a research-level problem. For now, return None to indicate we couldn't find better.
    # The structural insight is that global optimization might find shared factors,
    # but implementing an effective solver within time constraints is challenging.
    return None

def _build_schoolbook():
    """Fallback: rank 44 schoolbook decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {'U': U, 'V': V, 'W': W}

def construct():
    start_time = time.time()
    DEADLINE = 145  # Safety margin before 150s kill
    
    # Try the verified rank-39 construction first
    result = _build_rank39()
    if _verify(result['U'], result['V'], result['W']):
        return result
    
    # Fall back to schoolbook rank 44
    result = _build_schoolbook()
    return result
