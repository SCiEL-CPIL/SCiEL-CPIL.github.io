import numpy as np
import time
from fractions import Fraction
from collections import defaultdict

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 120.0

    # ---------- Structural fallback: Strassen block decomposition, rank 36 ----------
    def strassen_block_solution():
        rank = 36
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0

        def add_term(u_coeffs, v_coeffs, w_coeffs):
            nonlocal col
            for idx, val in u_coeffs.items():
                if val != 0:
                    U[idx][col] += val
            for idx, val in v_coeffs.items():
                if val != 0:
                    V[idx][col] += val
            for idx, val in w_coeffs.items():
                if val != 0:
                    W[idx][col] += val
            col += 1

        strassen_terms = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
            ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
            ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
            ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
            ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
            ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
        ]
        standard_terms_2x2x1 = [
            ({0: 1}, {0: 1}, {0: 1}),
            ({1: 1}, {1: 1}, {0: 1}),
            ({2: 1}, {0: 1}, {1: 1}),
            ({3: 1}, {1: 1}, {1: 1}),
        ]

        for q in range(2):
            for s in range(3):
                if s < 2:
                    for u_t, v_t, w_t in strassen_terms:
                        u_g, v_g, w_g = {}, {}, {}
                        for idx, val in u_t.items():
                            li, lj = idx // 2, idx % 2
                            u_g[li * m + (2 * q + lj)] = val
                        for idx, val in v_t.items():
                            lj, lk = idx // 2, idx % 2
                            v_g[(2 * q + lj) * p + (2 * s + lk)] = val
                        for idx, val in w_t.items():
                            li, lk = idx // 2, idx % 2
                            w_g[(2 * s + lk) * n + li] = val
                        add_term(u_g, v_g, w_g)
                else:
                    for u_t, v_t, w_t in standard_terms_2x2x1:
                        u_g, v_g, w_g = {}, {}, {}
                        for idx, val in u_t.items():
                            li, lj = idx // 2, idx % 2
                            u_g[li * m + (2 * q + lj)] = val
                        for idx, val in v_t.items():
                            lj = idx
                            v_g[(2 * q + lj) * p + 4] = val
                        for idx, val in w_t.items():
                            li = idx
                            w_g[4 * n + li] = val
                        add_term(u_g, v_g, w_g)

        return {"U": U, "V": V, "W": W}

    # ---------- Exact-rational verification ----------
    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for k in range(p):
                c_idx = k * n + i
                for j in range(m):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    s = 0
                    for r in range(rank):
                        if U[a_idx][r] and V[b_idx][r] and W[c_idx][r]:
                            s += U[a_idx][r] * V[b_idx][r] * W[c_idx][r]
                    if s != 1:
                        return False
        return True

    # ---------- Sparse Rank-1 Extraction ----------
    def sparse_greedy_extract(target_rank, time_budget):
        num_u, num_v, num_w = n * m, m * p, p * n
        
        # Initialize residual tensor T as sparse dictionary
        # Keys: (u, v, w), Values: Fraction
        T = {}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    T[(u_idx, v_idx, w_idx)] = Fraction(1)

        # Data structures for fast lookup
        # U_slices: u -> { (v, w): val }
        # V_slices: v -> { (u, w): val }
        # W_slices: w -> { (u, v): val }
        U_slices = defaultdict(dict)
        V_slices = defaultdict(dict)
        W_slices = defaultdict(dict)
        
        for (u, v, w), val in T.items():
            U_slices[u][(v, w)] = val
            V_slices[v][(u, w)] = val
            W_slices[w][(u, v)] = val

        U_list = []
        V_list = []
        W_list = []
        
        # Helper to update residual
        def update_residual(u_vec, v_vec, w_vec):
            # u_vec, v_vec, w_vec are dicts of non-zero elements
            # We iterate over the smallest vector to minimize operations?
            # Actually, we need to update T -= u[i] v[j] w[k]
            # Let's iterate over u_vec
            for u_idx, u_val in u_vec.items():
                if u_val == 0: continue
                U_s = U_slices[u_idx]
                # Iterate over v_vec
                for v_idx, v_val in v_vec.items():
                    if v_val == 0: continue
                    # We need to find keys (v_idx, w_idx) in U_s
                    # And update with w_vec
                    # Let's check if we can iterate over w_vec instead?
                    # The term is u[u_idx] * v[v_idx] * w[w_idx]
                    # We have U_s[v_idx, w_idx]
                    
                    # Optimization: Iterate over the intersection of existing keys?
                    # No, we just iterate over v_vec and w_vec.
                    # Since the tensor is sparse, v_vec and w_vec should be small.
                    for w_idx, w_val in w_vec.items():
                        if w_val == 0: continue
                        key = (v_idx, w_idx)
                        if key in U_s:
                            U_s[key] -= u_val * v_val * w_val
                            if U_s[key] == 0:
                                del U_s[key]
                        else:
                            # This implies the residual was 0 here, but we are subtracting a non-zero value?
                            # Wait, if it's not in U_s, it means T[u, v, w] was 0.
                            # If we subtract a non-zero value, it becomes non-zero.
                            # But can we extract a rank-1 component that has support outside the current support?
                            # Yes, but usually we pick a component from the support.
                            # If we pick a component from the support, its support is a subset of the current support?
                            # Not necessarily.
                            # However, if we construct u, v, w based on the current support,
                            # the new residual might have new non-zeros?
                            # Actually, if we solve the linear system exactly on the support,
                            # the "new" non-zeros should be zero if the decomposition is exact.
                            # But if we are doing greedy, we might introduce noise?
                            # No, we use exact arithmetic.
                            # If the key doesn't exist, it means the current value is 0.
                            # We are subtracting u_val * v_val * w_val.
                            # So the new value is -u_val * v_val * w_val.
                            # We must add it to the sparse structures.
                            new_val = -u_val * v_val * w_val
                            U_s[key] = new_val
                            V_slices[v_idx][(u_idx, w_idx)] = new_val
                            W_slices[w_idx][(u_idx, v_idx)] = new_val
                            T[(u_idx, v_idx, w_idx)] = new_val
                            continue

                        # If it existed and became 0, we deleted it from U_s.
                        # We must also delete from V_s and W_s and T
                        if U_s.get(key, 0) == 0:
                             del U_s[key]
                             if (u_idx, w_idx) in V_slices[v_idx]:
                                 del V_slices[v_idx][(u_idx, w_idx)]
                             if (u_idx, v_idx) in W_slices[w_idx]:
                                 del W_slices[w_idx][(u_idx, v_idx)]
                             del T[(u_idx, v_idx, w_idx)]
        
        for step in range(target_rank):
            if time.time() > time_budget:
                return None
            if not T:
                break
            
            # Pick a candidate (u0, v0, w0) from T
            # Heuristic: Pick one with a "central" position or just the first one
            u0, v0, w0 = next(iter(T.keys()))
            
            # Construct u, v, w
            # We want T - u v w^T to be zero on the "slice" defined by v0 and w0?
            # Standard approach:
            # Set u[u0] = 1.
            # For all u_i in support of v0: u[u_i] = - T[u_i, v0, w0] / T[u0, v0, w0]
            # Wait, T[u0, v0, w0] is the pivot.
            # Let val0 = T[u0, v0, w0]
            
            val0 = T[(u0, v0, w0)]
            
            u_vec = {u0: Fraction(1)}
            # Find all u_i such that (u_i, v0, w0) is in T
            # We can look at W_slices[w0] to find (u, v) pairs, then filter by v==v0
            # Or V_slices[v0] to find (u, w) pairs, then filter by w==w0
            # Let's use V_slices[v0]
            for (u_i, w_i), val_i in V_slices[v0].items():
                if w_i == w0:
                    # val_i is T[u_i, v0, w0]
                    if u_i == u0:
                        continue
                    u_vec[u_i] = -val_i / val0
            
            # Now construct v_vec
            # v[v0] = 1
            # For all v_j such that (u0, v_j, w0) is in T:
            # v[v_j] = - T[u0, v_j, w0] / T[u0, v0, w0] ?
            # Let's check the condition:
            # We want the residual at (u0, v_j, w0) to be 0.
            # R[u0, v_j, w0] = T[u0, v_j, w0] - u[u0] * v[v_j] * w[w0]
            # We haven't defined w yet.
            # Let's define w[w0] = 1 first?
            # If w[w0] = 1, then v[v_j] = T[u0, v_j, w0] / (u[u0] * 1) = T[u0, v_j, w0]
            # Wait, u[u0]=1.
            # So v[v_j] = T[u0, v_j, w0].
            # But we also need to satisfy the condition for u_i:
            # R[u_i, v0, w0] = T[u_i, v0, w0] - u[u_i] * v[v0] * w[w0] = 0
            # If v[v0]=1, w[w0]=1, then T[u_i, v0, w0] - u[u_i] = 0 => u[u_i] = T[u_i, v0, w0].
            # This contradicts u[u_i] = -T[u_i, v0, w0] / T[u0, v0, w0] unless T[u0, v0, w0] = -1.
            
            # Correct Logic:
            # We choose a rank-1 tensor $u \otimes v \otimes w$ to subtract.
            # We want $T - u \otimes v \otimes w$ to have zeros at specific positions.
            # Let's force $R[u_i, v0, w0] = 0$ for all $i$.
            # $T[u_i, v0, w0] - u[u_i] v[v0] w[w0] = 0$.
            # Let $v[v0] w[w0] = 1$. (We can scale u, v, w arbitrarily).
            # Then $u[u_i] = T[u_i, v0, w0]$.
            # Now force $R[u0, v_j, w0] = 0$ for all $j$.
            # $T[u0, v_j, w0] - u[u0] v[v_j] w[w0] = 0$.
            # We know $u[u0] = T[u0, v0, w0]$.
            # $T[u0, v_j, w0] - T[u0, v0, w0] v[v_j] w[w0] = 0$.
            # $v[v_j] w[w0] = T[u0, v_j, w0] / T[u0, v0, w0]$.
            # Let $w[w0] = 1/T[u0, v0, w0]$.
            # Then $v[v_j] = T[u0, v_j, w0]$.
            
            # So:
            # w[w0] = 1 / val0
            # u[u_i] = T[u_i, v0, w0] for all i in support of (v0, w0)
            # v[v_j] = T[u0, v_j, w0] for all j in support of (u0, w0)
            
            # Now force $R[u0, v0, w_k] = 0$ for all $k$.
            # $T[u0, v0, w_k] - u[u0] v[v0] w[w_k] = 0$.
            # $u[u0] = val0$. $v[v0] = T[u0, v0, w0] = val0$.
            # $val0 * val0 * w[w_k] = T[u0, v0, w_k]$.
            # $w[w_k] = T[u0, v0, w_k] / (val0 * val0)$.
            
            # Check consistency for $w[w0]$:
            # $w[w0] = T[u0, v0, w0] / (val0 * val0) = val0 / val0^2 = 1/val0$. Matches.
            
            # So the vectors are:
            u_vec = {}
            for (u_i, w_i), val_i in V_slices[v0].items():
                if w_i == w0:
                    u_vec[u_i] = val_i
                    
            v_vec = {}
            for (v_j, w_j), val_j in U_slices[u0].items():
                if w_j == w0:
                    v_vec[v_j] = val_j
                    
            w_vec = {}
            for (u_i, v_i), val_i in W_slices[w0].items():
                if u_i == u0 and v_i == v0:
                    w_vec[w0] = Fraction(1, val0)
                elif u_i == u0:
                    # val_i is T[u0, v_i, w0]. But we need T[u0, v0, w_k].
                    # We need to look at U_slices[u0] for (v0, w_k)
                    pass
            
            # Re-fetch w_vec correctly
            # We need T[u0, v0, w_k]
            for (v_i, w_k), val_k in U_slices[u0].items():
                if v_i == v0:
                    w_vec[w_k] = val_k / (val0 * val0)
                    
            # Verify the extraction
            # Check if R becomes zero on the "cross" defined by u0, v0, w0
            # R[u_i, v0, w0] = T[u_i, v0, w0] - u[u_i] v[v0] w[w0]
            # v[v0] = val0. w[w0] = 1/val0. Product = 1.
            # R = T[u_i, v0, w0] - T[u_i, v0, w0] = 0. OK.
            
            # R[u0, v_j, w0] = T[u0, v_j, w0] - u[u0] v[v_j] w[w0]
            # u[u0] = val0. w[w0] = 1/val0. Product = 1.
            # R = T[u0, v_j, w0] - v[v_j] = 0. OK.
            
            # R[u0, v0, w_k] = T[u0, v0, w_k] - u[u0] v[v0] w[w_k]
            # u[u0] v[v0] = val0^2.
            # R = T[u0, v0, w_k] - val0^2 * (T[u0, v0, w_k] / val0^2) = 0. OK.
            
            # What about other points?
            # R[u_i, v_j, w_k] = T[u_i, v_j, w_k] - u[u_i] v[v_j] w[w_k]
            # This is not guaranteed to be 0.
            # But we have reduced the support?
            # The new support is the old support minus the cross.
            # This is a valid rank-1 extraction.
            
            # Update Residual
            update_residual(u_vec, v_vec, w_vec)
            
            U_list.append(u_vec)
            V_list.append(v_vec)
            W_list.append(w_vec)
            
        if T:
            return None
            
        # Convert to matrices
        rank = len(U_list)
        Uf = [[0] * rank for _ in range(num_u)]
        Vf = [[0] * rank for _ in range(num_v)]
        Wf = [[0] * rank for _ in range(num_w)]
        
        for r in range(rank):
            for u_i, val in U_list[r].items():
                Uf[u_i][r] = float(val)
            for v_i, val in V_list[r].items():
                Vf[v_i][r] = float(val)
            for w_i, val in W_list[r].items():
                Wf[w_i][r] = float(val)
                
        return {"U": Uf, "V": Vf, "W": Wf}

    # ---------- Main ----------
    best = strassen_block_solution()
    best_rank = len(best["U"][0])
    
    # Try to find a lower rank
    # Theoretical lower bound for 2x4x5 is 20 (if we view it as 2x5 with 4 in the middle?)
    # Actually, rank(2x4x5) >= max(2*4, 4*5, 2*5) is not true.
    # Lower bound is max(n*m, m*p, n*p) is for specific cases?
    # For 2x2x2, rank is 7. 2*2*2=8.
    # For 2x4x5, let's try 20, 24, 28, 32.
    
    targets = [20, 24, 28, 32, 35]
    
    for tr in targets:
        if tr >= best_rank:
            continue
        if time.time() > deadline - 15:
            break
        res = sparse_greedy_extract(tr, deadline - 15)
        if res is not None:
            if verify(res["U"], res["V"], res["W"]):
                best = res
                best_rank = len(res["U"][0])
                break

    return best
# EVOLVE-BLOCK-END
