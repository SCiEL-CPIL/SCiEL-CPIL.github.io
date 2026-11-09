import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 3]
TIME_LIMIT = 115.0
START_TIME = time.time()


def elapsed():
    return time.time() - START_TIME


def deadline_exceeded():
    return elapsed() > TIME_LIMIT


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _verify_exact(U, V, W, n, m, p):
    R = len(U[0])
    Uf = [[Fraction(x) for x in row] for row in U]
    Vf = [[Fraction(x) for x in row] for row in V]
    Wf = [[Fraction(x) for x in row] for row in W]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                ui = Uf[i * m + j]
                vk = Vf[j * p + k]
                wk = Wf[k * n + i]
                for r in range(R):
                    s += ui[r] * vk[r] * wk[r]
                if s != 1:
                    return False
    return True


def _construct_rank17(n, m, p):
    """
    Constructive rank 17 decomposition for 2x3x3 matrix multiplication.
    Uses Strassen's algorithm for the 2x2x2 core (indices i,j,k in {0,1})
    and schoolbook terms for the remaining boundary indices (j=2 or k=2).
    """
    if n != 2 or m != 3 or p != 3:
        return None

    R = 17
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]

    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i

    # Strassen's 7 terms for i,j,k in {0,1}
    # r=0 (M1)
    U[u_idx(0,0)][0] = 1; U[u_idx(1,1)][0] = 1
    V[v_idx(0,0)][0] = 1; V[v_idx(1,1)][0] = 1
    W[w_idx(0,0)][0] = 1; W[w_idx(1,1)][0] = 1
    
    # r=1 (M2)
    U[u_idx(1,0)][1] = 1; U[u_idx(1,1)][1] = 1
    V[v_idx(0,0)][1] = 1
    W[w_idx(0,1)][1] = 1; W[w_idx(1,1)][1] = -1
    
    # r=2 (M3)
    U[u_idx(0,0)][2] = 1
    V[v_idx(0,1)][2] = 1; V[v_idx(1,1)][2] = -1
    W[w_idx(1,0)][2] = 1; W[w_idx(1,1)][2] = 1
    
    # r=3 (M4)
    U[u_idx(1,1)][3] = 1
    V[v_idx(1,0)][3] = 1; V[v_idx(0,0)][3] = -1
    W[w_idx(0,0)][3] = 1; W[w_idx(0,1)][3] = 1
    
    # r=4 (M5)
    U[u_idx(0,0)][4] = 1; U[u_idx(0,1)][4] = 1
    V[v_idx(1,1)][4] = 1
    W[w_idx(0,0)][4] = -1; W[w_idx(1,0)][4] = 1
    
    # r=5 (M6)
    U[u_idx(1,0)][5] = 1; U[u_idx(0,0)][5] = -1
    V[v_idx(0,0)][5] = 1; V[v_idx(0,1)][5] = 1
    W[w_idx(1,1)][5] = 1
    
    # r=6 (M7)
    U[u_idx(0,1)][6] = 1; U[u_idx(1,1)][6] = -1
    V[v_idx(1,0)][6] = 1; V[v_idx(1,1)][6] = 1
    W[w_idx(0,0)][6] = 1

    # Boundary terms (r=7..16)
    boundary_terms = [
        (0, 2, 0), (1, 2, 0),
        (0, 2, 1), (1, 2, 1),
        (0, 2, 2), (1, 2, 2),
        (0, 0, 2), (1, 0, 2),
        (0, 1, 2), (1, 1, 2)
    ]
    
    for idx, (i, j, k) in enumerate(boundary_terms):
        r = 7 + idx
        U[u_idx(i, j)][r] = 1
        V[v_idx(j, k)][r] = 1
        W[w_idx(k, i)][r] = 1

    return {"U": U, "V": V, "W": W}


def _try_reduce_rank(U, V, W, n, m, p, target_rank, max_time=30.0):
    """
    Try to reduce rank by checking if any rank-1 tensor component is linearly
    dependent on the others. Uses Gaussian elimination over Q.
    """
    current_rank = len(U[0])
    if current_rank <= target_rank:
        return None
    
    Uf = [[Fraction(x) for x in row] for row in U]
    Vf = [[Fraction(x) for x in row] for row in V]
    Wf = [[Fraction(x) for x in row] for row in W]
    
    start_time = time.time()
    current_U = [row[:] for row in Uf]
    current_V = [row[:] for row in Vf]
    current_W = [row[:] for row in Wf]
    current_R = current_rank
    
    # Try to eliminate tensors one at a time
    for attempt in range(current_R - target_rank):
        if time.time() - start_time > max_time or deadline_exceeded():
            break
        if current_R <= target_rank:
            break
        
        eliminated = False
        
        # Try each tensor as the one to eliminate
        for elim_idx in range(current_R):
            if time.time() - start_time > max_time or deadline_exceeded():
                break
            
            # Set up: can tensor elim_idx be expressed as linear combo of others?
            # For each (i,j,k): sum_{r != elim_idx} c_r * U[i,j,r]*V[j,k,r]*W[k,i,r] 
            #   = U[i,j,elim_idx]*V[j,k,elim_idx]*W[k,i,elim_idx]
            
            others = [r for r in range(current_R) if r != elim_idx]
            num_unknowns = len(others)
            num_eqs = n * m * p
            
            # Build augmented matrix
            aug = []
            for (i, j, k) in product(range(n), range(m), range(p)):
                row = []
                for r in others:
                    row.append(current_U[i*m+j][r] * current_V[j*p+k][r] * current_W[k*n+i][r])
                rhs = current_U[i*m+j][elim_idx] * current_V[j*p+k][elim_idx] * current_W[k*n+i][elim_idx]
                aug.append(row + [rhs])
            
            # Gaussian elimination
            rank = 0
            for col in range(num_unknowns):
                if rank >= num_eqs:
                    break
                pivot = None
                for row in range(rank, num_eqs):
                    if aug[row][col] != 0:
                        pivot = row
                        break
                if pivot is None:
                    continue
                aug[rank], aug[pivot] = aug[pivot], aug[rank]
                pivot_val = aug[rank][col]
                for row in range(num_eqs):
                    if row != rank and aug[row][col] != 0:
                        factor = aug[row][col] / pivot_val
                        for c in range(col, num_unknowns + 1):
                            aug[row][c] -= factor * aug[rank][c]
                rank += 1
            
            # Check consistency
            consistent = True
            for row in range(rank, num_eqs):
                if aug[row][num_unknowns] != 0:
                    consistent = False
                    break
            
            if not consistent:
                continue
            
            # Extract solution
            coeffs = [Fraction(0)] * num_unknowns
            for row in range(rank):
                lead_col = None
                for c in range(num_unknowns):
                    if aug[row][c] != 0:
                        lead_col = c
                        break
                if lead_col is not None:
                    coeffs[lead_col] = aug[row][num_unknowns] / aug[row][lead_col]
            
            # Verify
            valid = True
            for (i, j, k) in product(range(n), range(m), range(p)):
                lhs = Fraction(0)
                for idx2, r in enumerate(others):
                    lhs += coeffs[idx2] * current_U[i*m+j][r] * current_V[j*p+k][r] * current_W[k*n+i][r]
                rhs = current_U[i*m+j][elim_idx] * current_V[j*p+k][elim_idx] * current_W[k*n+i][elim_idx]
                if lhs != rhs:
                    valid = False
                    break
            
            if valid:
                # Apply elimination: subtract coeffs[r]*U[elim] from U[r] for each r in others
                for idx2, r in enumerate(others):
                    if coeffs[idx2] != 0:
                        c = coeffs[idx2]
                        for i in range(n * m):
                            current_U[i][r] -= c * current_U[i][elim_idx]
                        for j in range(m * p):
                            current_V[j][r] -= c * current_V[j][elim_idx]
                        for k in range(p * n):
                            current_W[k][r] -= c * current_W[k][elim_idx]
                
                # Remove column elim_idx
                for i in range(n * m):
                    current_U[i].pop(elim_idx)
                for j in range(m * p):
                    current_V[j].pop(elim_idx)
                for k in range(p * n):
                    current_W[k].pop(elim_idx)
                current_R -= 1
                eliminated = True
                break
        
        if not eliminated:
            break
    
    if current_R <= target_rank:
        U_out = [[int(x) if x.denominator == 1 else float(x) for x in row] for row in current_U]
        V_out = [[int(x) if x.denominator == 1 else float(x) for x in row] for row in current_V]
        W_out = [[int(x) if x.denominator == 1 else float(x) for x in row] for row in current_W]
        return {"U": U_out, "V": V_out, "W": W_out}
    
    return None


def _try_rank16_construction(n, m, p):
    """
    Try to construct a rank-16 decomposition directly.
    Strategy: Use Strassen for the 2x2x2 core (7 terms), and try to cover
    the 10 boundary terms with only 9 rank-1 tensors using clever combinations.
    
    The boundary terms are:
    - Group A: (0,2,0), (1,2,0) -> need A[i,2]*B[2,0] for i=0,1 -> 2 terms (rank 2 for 2x1x1)
    - Group B: (0,2,1), (1,2,1) -> need A[i,2]*B[2,1] for i=0,1 -> 2 terms
    - Group C: (0,2,2), (1,2,2) -> need A[i,2]*B[2,2] for i=0,1 -> 2 terms  
    - Group D: (0,0,2), (1,0,2) -> need A[i,0]*B[0,2] for i=0,1 -> 2 terms
    - Group E: (0,1,2), (1,1,2) -> need A[i,1]*B[1,2] for i=0,1 -> 2 terms
    
    Total boundary: 10 terms. Can we do it in 9?
    
    Actually, groups A, B, C share the same A column (j=2) but different B rows.
    Groups D, E share B column (k=2) but different A columns.
    
    Let's try: For groups A, B, C (all j=2), we have a 2x1x3 block (A rows 0,1; B col 2; C cols 0,1,2).
    This is a 2x1x3 tensor. Its rank is 2 (since it's outer product of [A02, A12] with [B20, B21, B22]).
    So we need only 2 rank-1 tensors for all of j=2!
    
    Similarly for groups D, E (all k=2), we have a 2x2x1 block (A cols 0,1; B row 2; C row 2).
    This is rank 2 as well.
    
    But wait - the C indices overlap! Group C gives C[2,0] and C[2,1], and groups D,E also give C[2,0] and C[2,1].
    So we need to be careful about the overlap.
    
    Let me reconsider. The full boundary set is:
    j=2: (i,2,k) for i in {0,1}, k in {0,1,2} -> 6 terms
    k=2, j<2: (i,j,2) for i in {0,1}, j in {0,1} -> 4 terms
    Total: 10 terms
    
    For j=2 block: C[k,i] gets contributions from A[i,2]*B[2,k]
    For k=2, j<2 block: C[2,i] gets contributions from A[i,j]*B[j,2]
    
    The overlap is C[2,i] which appears in both blocks.
    
    Let's try:
    - 2 rank-1 tensors for j=2 block: these give C[k,i] for k=0,1,2
    - 2 rank-1 tensors for k=2, j<2 block: these give C[2,i] for i=0,1
    
    But the C[2,i] entries will get contributions from BOTH sets, so we need to subtract.
    
    Actually, let's just try a direct rank-16 construction with specific coefficients.
    """
    if n != 2 or m != 3 or p != 3:
        return None
    
    # Try rank 16 with modified boundary handling
    R = 16
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    # Strassen's 7 terms for i,j,k in {0,1}
    U[u_idx(0,0)][0] = 1; U[u_idx(1,1)][0] = 1
    V[v_idx(0,0)][0] = 1; V[v_idx(1,1)][0] = 1
    W[w_idx(0,0)][0] = 1; W[w_idx(1,1)][0] = 1
    
    U[u_idx(1,0)][1] = 1; U[u_idx(1,1)][1] = 1
    V[v_idx(0,0)][1] = 1
    W[w_idx(0,1)][1] = 1; W[w_idx(1,1)][1] = -1
    
    U[u_idx(0,0)][2] = 1
    V[v_idx(0,1)][2] = 1; V[v_idx(1,1)][2] = -1
    W[w_idx(1,0)][2] = 1; W[w_idx(1,1)][2] = 1
    
    U[u_idx(1,1)][3] = 1
    V[v_idx(1,0)][3] = 1; V[v_idx(0,0)][3] = -1
    W[w_idx(0,0)][3] = 1; W[w_idx(0,1)][3] = 1
    
    U[u_idx(0,0)][4] = 1; U[u_idx(0,1)][4] = 1
    V[v_idx(1,1)][4] = 1
    W[w_idx(0,0)][4] = -1; W[w_idx(1,0)][4] = 1
    
    U[u_idx(1,0)][5] = 1; U[u_idx(0,0)][5] = -1
    V[v_idx(0,0)][5] = 1; V[v_idx(0,1)][5] = 1
    W[w_idx(1,1)][5] = 1
    
    U[u_idx(0,1)][6] = 1; U[u_idx(1,1)][6] = -1
    V[v_idx(1,0)][6] = 1; V[v_idx(1,1)][6] = 1
    W[w_idx(0,0)][6] = 1
    
    # For boundary, use 9 terms instead of 10
    # Strategy: merge some boundary terms using Strassen-like tricks
    
    # Boundary terms we need:
    # (0,2,0), (1,2,0), (0,2,1), (1,2,1), (0,2,2), (1,2,2), (0,0,2), (1,0,2), (0,1,2), (1,1,2)
    
    # Try: use rank-2 decomposition for the j=2 block (6 terms) and rank-2 for k=2,j<2 block (4 terms)
    # but with correction for overlap
    
    # j=2 block: A[i,2]*B[2,k] for i in {0,1}, k in {0,1,2}
    # Use: M_a = A[0,2]*B[2,0] + A[1,2]*B[2,1]  (diagonal-like)
    # M_b = (A[0,2]+A[1,2])*(B[2,0]+B[2,1])  (sum)
    # Then: A[0,2]*B[2,1] = M_b - M_a - A[1,2]*B[2,0]
    # This doesn't quite work cleanly...
    
    # Let me just try specific rank-16 patterns and verify
    # Pattern 1: merge (0,2,0) and (1,2,1) into one term, etc.
    
    # Actually, let's try: for the 4 terms (0,2,0),(1,2,0),(0,2,1),(1,2,1), use Strassen-like:
    # These form a 2x1x2 block. Rank of 2x1x2 is 2 (it's outer product).
    # So: T1 = A[:,2] * B[2,0], T2 = A[:,2] * B[2,1] -> 2 terms for 4 equations
    
    # Similarly (0,2,2),(1,2,2) -> 1 term: T3 = A[:,2] * B[2,2]
    # (0,0,2),(1,0,2) -> 1 term: T4 = A[:,0] * B[0,2]  
    # (0,1,2),(1,1,2) -> 1 term: T5 = A[:,1] * B[1,2]
    
    # Total boundary: 2+1+1+1 = 5 terms? No, that's not right.
    # T1 covers (0,2,0) and (1,2,0) -> 2 equations
    # T2 covers (0,2,1) and (1,2,1) -> 2 equations
    # T3 covers (0,2,2) and (1,2,2) -> 2 equations
    # T4 covers (0,0,2) and (1,0,2) -> 2 equations
    # T5 covers (0,1,2) and (1,1,2) -> 2 equations
    # Total: 5 terms for 10 equations. That's 5 boundary terms!
    # Plus 7 Strassen = 12 total? That can't be right...
    
    # Wait, I'm confusing myself. Each rank-1 term U_r * V_r * W_r contributes to ALL (i,j,k).
    # T1 = A[:,2]*B[2,0] means: U[0,2]=1, U[1,2]=1, V[2,0]=1, W[0,0]=1, W[0,1]=1
    # This contributes to (0,2,0) and (1,2,0) in C.
    
    # So yes, 5 boundary terms should suffice! But then total rank = 7+5 = 12?
    # That seems too good. Let me check if these 5 terms are linearly independent from the Strassen terms.
    
    # The issue: the Strassen terms already have non-zero entries in some of these positions.
    # For example, Strassen term M1 has U[0,0]=1, U[1,1]=1, V[0,0]=1, V[1,1]=1, W[0,0]=1, W[1,1]=1
    # This contributes to (0,0,0) and (1,1,1) only.
    
    # The boundary terms T1-T5 have non-zero entries only in positions where j=2 or k=2.
    # The Strassen terms have non-zero entries only where i,j,k in {0,1}.
    # So they are disjoint! The 5 boundary terms are independent of the 7 Strassen terms.
    
    # So rank 12 should work?? Let me verify this claim.
    
    # T1: U[0,2]=1, U[1,2]=1, V[2,0]=1, W[0,0]=1, W[0,1]=1
    # Contribution: (0,2,0): 1*1*1=1, (1,2,0): 1*1*1=1. Good.
    # But also: (0,2,0) in C is C[0,0], (1,2,0) is C[0,1].
    # Wait, W[k*n+i]: W[0*2+0]=W[0,0], W[0*2+1]=W[0,1]. Yes.
    
    # T2: U[0,2]=1, U[1,2]=1, V[2,1]=1, W[1,0]=1, W[1,1]=1
    # Contribution: (0,2,1): C[1,0], (1,2,1): C[1,1]. Good.
    
    # T3: U[0,2]=1, U[1,2]=1, V[2,2]=1, W[2,0]=1, W[2,1]=1
    # Contribution: (0,2,2): C[2,0], (1,2,2): C[2,1]. Good.
    
    # T4: U[0,0]=1, U[1,0]=1, V[0,2]=1, W[2,0]=1, W[2,1]=1
    # Contribution: (0,0,2): C[2,0], (1,0,2): C[2,1]. Good.
    
    # T5: U[0,1]=1, U[1,1]=1, V[1,2]=1, W[2,0]=1, W[2,1]=1
    # Contribution: (0,1,2): C[2,0], (1,1,2): C[2,1]. Good.
    
    # Now check: C[2,0] gets contributions from T3, T4, T5: 1+1+1=3. Should be 1!
    # C[2,1] gets contributions from T3, T4, T5: 1+1+1=3. Should be 1!
    
    # So we need to fix the overlap. C[2,i] should be 1, but we're getting 3.
    # We need: T3_W + T4_W + T5_W = 1 for each i.
    # Currently each is 1, so sum is 3.
    # Solution: use coefficients. Let T3_W = a, T4_W = b, T5_W = c with a+b+c=1.
    # But also T3, T4, T5 need to give correct contributions to their respective (i,j,k).
    
    # For T3: contributes to (0,2,2) and (1,2,2). C[2,0] gets a, C[2,1] gets a.
    # For T4: contributes to (0,0,2) and (1,0,2). C[2,0] gets b, C[2,1] gets b.
    # For T5: contributes to (0,1,2) and (1,1,2). C[2,0] gets c, C[2,1] gets c.
    
    # We need a=1, b=1, c=1 for the individual (i,j,k) to be correct, but a+b+c=3≠1.
    # Contradiction! So we can't do it with just 5 boundary terms.
    
    # We need at least 6 boundary terms to handle the overlap properly.
    # Or use a different decomposition.
    
    # Let me try: keep 10 boundary terms but see if any can be merged with Strassen terms.
    # Since the supports are disjoint, they can't be merged. So rank 17 is the minimum
    # with this construction.
    
    # But maybe a completely different construction achieves rank 16 or lower.
    # For now, return None and let the reduction algorithm try.
    
    return None


def construct():
    n, m, p = DIMENSIONS
    
    # Try constructive rank 17
    result = _construct_rank17(n, m, p)
    if result is not None:
        if _verify_exact(result["U"], result["V"], result["W"], n, m, p):
            # Try to reduce rank further
            for target in range(16, 10, -1):
                if deadline_exceeded():
                    break
                reduced = _try_reduce_rank(result["U"], result["V"], result["W"], n, m, p, target, max_time=25.0)
                if reduced is not None:
                    if _verify_exact(reduced["U"], reduced["V"], reduced["W"], n, m, p):
                        result = reduced
                        break
    
    if result is None:
        result = _schoolbook(n, m, p)
    
    return result


def main():
    return construct()


if __name__ == '__main__':
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
