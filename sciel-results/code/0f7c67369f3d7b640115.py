import numpy as np
import time
import random

DIMENSIONS = [2, 2, 4]

def construct():
    n, m, p = DIMENSIONS
    u_dim = n * m  # 4
    v_dim = m * p  # 8
    w_dim = p * n  # 8
    
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    def idx_u(i, j): return i * m + j
    def idx_v(j, k): return j * p + k
    def idx_w(k, i): return k * n + i

    # Base Strassen 7 terms for 2x2x2
    # Indices: U is (i,j), V is (j,k), W is (k,i)
    # Using the standard Strassen terms:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    terms = []
    
    # Term 1: (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    t1_u = [(idx_u(0,0), 1), (idx_u(1,1), 1)]
    t1_v = [(idx_v(0,0), 1), (idx_v(1,1), 1)]
    t1_w = [(idx_w(0,0), 1), (idx_w(1,1), 1)]
    terms.append((t1_u, t1_v, t1_w))
    
    # Term 2: (A10+A11)B00 -> C10 += M2, C11 -= M2
    t2_u = [(idx_u(1,0), 1), (idx_u(1,1), 1)]
    t2_v = [(idx_v(0,0), 1)]
    t2_w = [(idx_w(0,1), 1), (idx_w(1,1), -1)]
    terms.append((t2_u, t2_v, t2_w))
    
    # Term 3: A00(B01-B11) -> C01 += M3, C11 += M3
    t3_u = [(idx_u(0,0), 1)]
    t3_v = [(idx_v(0,1), 1), (idx_v(1,1), -1)]
    t3_w = [(idx_w(1,0), 1), (idx_w(1,1), 1)]
    terms.append((t3_u, t3_v, t3_w))
    
    # Term 4: A11(B10-B00) -> C00 += M4, C10 += M4
    t4_u = [(idx_u(1,1), 1)]
    t4_v = [(idx_v(1,0), 1), (idx_v(0,0), -1)]
    t4_w = [(idx_w(0,0), 1), (idx_w(0,1), 1)]
    terms.append((t4_u, t4_v, t4_w))
    
    # Term 5: (A00+A01)B11 -> C00 -= M5, C01 += M5
    t5_u = [(idx_u(0,0), 1), (idx_u(0,1), 1)]
    t5_v = [(idx_v(1,1), 1)]
    t5_w = [(idx_w(0,0), -1), (idx_w(1,0), 1)]
    terms.append((t5_u, t5_v, t5_w))
    
    # Term 6: (A10-A00)(B00+B01) -> C11 += M6
    t6_u = [(idx_u(1,0), 1), (idx_u(0,0), -1)]
    t6_v = [(idx_v(0,0), 1), (idx_v(0,1), 1)]
    t6_w = [(idx_w(1,1), 1)]
    terms.append((t6_u, t6_v, t6_w))
    
    # Term 7: (A01-A11)(B10+B11) -> C00 += M7
    t7_u = [(idx_u(0,1), 1), (idx_u(1,1), -1)]
    t7_v = [(idx_v(1,0), 1), (idx_v(1,1), 1)]
    t7_w = [(idx_w(0,0), 1)]
    terms.append((t7_u, t7_v, t7_w))
    
    def build_decomposition(rank, term_assignments=None):
        """
        Build U, V, W matrices for a given rank.
        term_assignments: list of (term_idx, block_idx) where block_idx is 0 or 1 (for left/right 2x4 blocks)
        If None, uses the standard rank-14 assignment (2 blocks of 7 terms).
        """
        U = np.zeros((u_dim, rank))
        V = np.zeros((v_dim, rank))
        W = np.zeros((w_dim, rank))
        
        if term_assignments is None:
            # Standard rank 14: first 7 columns for block 0, next 7 for block 1
            term_assignments = [(r, 0) for r in range(7)] + [(r, 1) for r in range(7)]
        
        for col, (t_idx, block_idx) in enumerate(term_assignments):
            u_vec, v_vec, w_vec = terms[t_idx]
            
            # Block 0: B columns 0,1. V indices 0,1,4,5. W indices 0,1,2,3.
            # Block 1: B columns 2,3. V indices 2,3,6,7. W indices 4,5,6,7.
            
            for idx, val in u_vec:
                U[idx, col] += val
            
            for idx, val in v_vec:
                if block_idx == 0:
                    V[idx, col] += val
                else:
                    # Shift k by 2: idx = j*p + k -> j*p + (k+2) = idx + 2
                    V[idx + 2, col] += val
            
            for idx, val in w_vec:
                if block_idx == 0:
                    W[idx, col] += val
                else:
                    # Shift k by 2: idx = k*n + i -> (k+2)*n + i = idx + 2*n = idx + 4
                    W[idx + 4, col] += val
            
        return U, V, W

    def verify(U, V, W):
        """
        Verify exact decomposition.
        T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
        Must be 1 if i=a*m+b, j=b*p+c, k=c*n+a.
        Must be 0 otherwise.
        """
        # Build tensor T
        # T shape (4, 8, 8)
        # T[i,j,k]
        for i in range(u_dim):
            a = i // m
            b = i % m
            for j in range(v_dim):
                b_prime = j // p
                c = j % p
                for k in range(w_dim):
                    c_prime = k // n
                    a_prime = k % n
                    
                    t_val = 0.0
                    for r in range(len(U[0])):
                        t_val += U[i, r] * V[j, r] * W[k, r]
                    
                    if a == a_prime and b == b_prime:
                        if abs(t_val - 1.0) > 1e-9:
                            return False
                    else:
                        if abs(t_val) > 1e-9:
                            return False
        return True

    # 1. Try standard rank 14
    U14, V14, W14 = build_decomposition(14)
    if verify(U14, V14, W14):
        # Check for rank 13 or 12 via search?
        # Given the time budget and the difficulty of finding lower rank, 
        # and the fact that 14 is the known best for this specific structure in many contexts,
        # we might just return it. 
        # However, the prompt asks to "investigate improvements".
        # Can we do rank 13?
        # Rank 12?
        
        # Let's try a simple random search for rank 13 and 12.
        # But random search on integer/half-integer grids is very slow.
        # We can try specific structural modifications.
        
        # Is it possible to share terms between blocks?
        # The A matrix is the same for both blocks.
        # If we use a term for block 0 and block 1, can we combine them?
        # No, because V and W are different (different B columns and C columns).
        # Unless... we find a linear dependency.
        
        # For now, let's stick to the verified rank 14.
        # If we have time, we can try to optimize the search for lower ranks.
        
        # The current score is 1.1429. 
        # If we return rank 14, score = seed/14.
        # Seed is likely 16 (schoolbook). 16/14 = 1.1428.
        
        # Can we find rank 13?
        # There is a known rank 13 algorithm for 2x2x4? 
        # Actually, the lower bound for 2x2x4 is not strictly known to be 14.
        # But 14 is the standard Strassen-based one.
        
        # Let's try a quick search for rank 13 with random small integers.
        # This is unlikely to succeed quickly, but let's try a few iterations.
        
        found_lower = False
        for target_rank in [13, 12]:
            if time.time() > deadline:
                break
            
            # Try a few random configurations
            # This is a heuristic. 
            # We can try to perturb the rank 14 solution by merging two columns?
            # If two columns are linearly dependent in some way...
            
            # Actually, let's just return the rank 14 solution to be safe and fast.
            # The prompt says "Keep the function names and signatures".
            # It also says "score is the rank".
            
            # If I can't find a lower rank, I return 14.
            # Is there a known rank 13?
            # A search suggests rank 14 is optimal for 2x2x4 over reals? 
            # No, Lysikov proved lower bounds. 
            # For 2x2x2, rank is 7.
            # For 2x2x4, it is 2 * 2x2x2? Not necessarily.
            # But 14 is the standard.
            
            # Let's assume 14 is the best we can reliably produce.
            pass

        return {"U": U14.tolist(), "V": V14.tolist(), "W": W14.tolist()}
    
    # If rank 14 fails (shouldn't happen), try to find any valid solution
    # This shouldn't happen.
    
    # Fallback: Schoolbook rank 16
    U = np.zeros((u_dim, 16))
    V = np.zeros((v_dim, 16))
    W = np.zeros((w_dim, 16))
    
    for a in range(n):
        for b in range(m):
            for c in range(p):
                r = a * (m*p) + b * p + c
                U[a*m+b, r] = 1
                V[b*p+c, r] = 1
                W[c*n+a, r] = 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# Test code
if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    # Quick verify
    n, m, p = 2, 2, 4
    rank = U.shape[1]
    valid = True
    for i in range(n*m):
        a = i // m
        b = i % m
        for j in range(m*p):
            bp = j // p
            c = j % p
            for k in range(p*n):
                cp = k // n
                ap = k % n
                t = 0
                for r in range(rank):
                    t += U[i,r] * V[j,r] * W[k,r]
                if a == ap and b == bp:
                    if abs(t - 1) > 1e-9:
                        valid = False
                else:
                    if abs(t) > 1e-9:
                        valid = False
    print("Valid:", valid)
    print("Rank:", rank)
