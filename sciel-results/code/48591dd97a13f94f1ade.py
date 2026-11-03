"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def solve(a, b):
    n = len(a)
    # 1. Initialization: Rank-matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Precompute some helpers if needed, but for n=150 direct calc is fine.
    # We will perform 2-opt local search.
    
    def compute_cost(p):
        # Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
        # Vectorized: (a * b[p][:, p]).sum()
        return np.sum(a * b[np.ix_(p, p)])
    
    current_cost = compute_cost(p)
    
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    # Local search loop
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # We can try to find the best swap in a pass
        best_delta = 0
        best_i = -1
        best_j = -1
        
        # To speed up, we can vectorize the delta calculation for all j for a given i.
        # Delta(i, j) formula derived earlier.
        # Let's stick to a simple loop first, n=150 is small.
        # If it's too slow, we will optimize.
        
        for i in range(n):
            if time.time() > deadline:
                break
            u = p[i]
            # For this i, calculate delta for all j > i
            # We can vectorize the sum over k for all j at once.
            
            # Terms:
            # 1. sum_{k != i, j} A[i, k] * (B[v, p[k]] - B[u, p[k]])
            # 2. sum_{k != i, j} A[k, i] * (B[p[k], v] - B[p[k], u])
            # 3. sum_{k != i, j} A[j, k] * (B[u, p[k]] - B[v, p[k]])
            # 4. sum_{k != i, j} A[k, j] * (B[p[k], u] - B[p[k], v])
            # 5. Interaction terms for i, j
            
            # Let's precompute row/col sums of A weighted by B rows/cols at current p?
            # This is complex. Let's just use numpy arrays for the inner sum.
            
            # For all j, we want delta.
            # v = p[j]
            
            # Let's create a matrix of B at current p: Bp = b[np.ix_(p, p)]
            # But p changes.
            
            # Simpler: just loop j and compute delta in O(1) if we maintain aggregates?
            # Or O(n) with numpy.
            
            # Let's try a hybrid: loop i, and for each i, compute deltas for all j using numpy.
            
            # Precompute for current p:
            # B_p = b[np.ix_(p, p)] is not directly useful for the delta formula which mixes p[k] and p[i], p[j].
            
            # Let's use the formula:
            # Delta(i, j) = 
            #   sum_k A[i,k] (B[p[j], p[k]] - B[p[i], p[k]])
            # + sum_k A[k,i] (B[p[k], p[j]] - B[p[k], p[i]])
            # + sum_k A[j,k] (B[p[i], p[k]] - B[p[j], p[k]])
            # + sum_k A[k,j] (B[p[k], p[i]] - B[p[k], p[j]])
            # + A[i,j] (B[p[j], p[i]] - B[p[i], p[j]])
            # + A[j,i] (B[p[i], p[j]] - B[p[j], p[i]])
            # Note: The sums include k=i and k=j, which needs correction or the formula above
            # already handles it if we are careful.
            # Let's check k=i in first term: A[i,i] (B[p[j], p[i]] - B[p[i], p[i]]).
            # In the original cost, term A[i,i] B[p[i], p[i]] exists.
            # In new cost, term A[i,i] B[p[j], p[j]]? No, p'(i)=p[j]. So A[i,i] B[p[j], p[j]].
            # Wait, the delta formula derived earlier was:
            # S'_ij - S_ij.
            
            # Let's just compute the full cost for the swapped permutation for the candidate best swap?
            # No, that's O(n^2) per swap.
            
            # Let's stick to the O(n) delta calculation inside the loop.
            # n=150, pairs=11175. O(n) per pair -> 1.6M ops per pass.
            # Python loop overhead might be high. 1.6M iterations in Python is ~0.1-0.2s.
            # So ~5-10 passes per second. 120s -> 600-1200 passes. Plenty.
            
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                
                u = p[i]
                v = p[j]
                
                # Calculate delta
                # We need sum over k != i, j.
                # We can compute sum over all k and subtract k=i and k=j terms.
                
                # Term 1: sum_k A[i,k] * (B[v, p[k]] - B[u, p[k]])
                # Note: B[v, p[k]] is b[v, p[k]].
                
                # Let's compute the sums using numpy vectors.
                # p is an array.
                
                # sum_k A[i,k] B[v, p[k]] - sum_k A[i,k] B[u, p[k]]
                # = A[i, :] @ b[v, p] - A[i, :] @ b[u, p]
                
                # Similarly for others.
                
                # Precompute b_rows_at_p: b[:, p] is n x n.
                # b_p = b[:, p]
                
                # delta = 
                # (A[i, :] @ b[v, p] - A[i, :] @ b[u, p]) 
                # + (A[:, i] @ b[:, v] - A[:, i] @ b[:, u])  # Note: b[:, v] is column v of b? No.
                # Wait, A[k,i] B[p[k], v]. Sum_k A[k,i] B[p[k], v].
                # This is A[:, i] @ b[p, v].
                
                # Let's be precise.
                # T1 = sum_k A[i,k] (B[v, p[k]] - B[u, p[k]])
                # T2 = sum_k A[k,i] (B[p[k], v] - B[p[k], u])
                # T3 = sum_k A[j,k] (B[u, p[k]] - B[v, p[k]])
                # T4 = sum_k A[k,j] (B[p[k], u] - B[p[k], v])
                # T5 = A[i,j] (B[v, u] - B[u, v])
                # T6 = A[j,i] (B[u, v] - B[v, u])
                
                # We must exclude k=i and k=j from T1..T4?
                # Let's check if the formula holds for all k.
                # Original Cost C = sum_{k,l} A[k,l] B[p[k], p[l]].
                # Change p[i]->v, p[j]->u.
                # Only terms with index i or j change.
                # C_old = sum_{k,l} A[k,l] B[p[k], p[l]]
                # C_new = C_old - (terms with i or j in old) + (terms with i or j in new)
                
                # Terms with i in old:
                # sum_l A[i,l] B[u, p[l]] + sum_k A[k,i] B[p[k], u]
                # Note: A[i,i] B[u,u] is in both sums.
                
                # Terms with j in old:
                # sum_l A[j,l] B[v, p[l]] + sum_k A[k,j] B[p[k], v]
                
                # Terms with i in new (p'[i]=v):
                # sum_l A[i,l] B[v, p'[l]] + sum_k A[k,i] B[p'[k], v]
                
                # Terms with j in new (p'[j]=u):
                # sum_l A[j,l] B[u, p'[l]] + sum_k A[k,j] B[p'[k], u]
                
                # This is getting complicated to vectorize correctly without double counting errors.
                # Given n=150, let's just compute the cost of the swapped permutation for the best candidate found in a pass?
                # No, we need to find the best swap.
                
                # Alternative: Just compute the delta using the O(n) formula, but implement it carefully.
                # Or, since n is small, just compute the full cost for a random set of swaps? No, we need systematic search.
                
                # Let's use the explicit delta formula with numpy dot products.
                # b_p = b[np.ix_(p, p)] is not needed.
                # We need:
                # v_row = b[v, p]  # shape (n,)
                # u_row = b[u, p]  # shape (n,)
                # v_col = b[p, v]  # shape (n,)
                # u_col = b[p, u]  # shape (n,)
                
                # T1 = A[i, :] @ (v_row - u_row)
                # T2 = A[:, i] @ (v_col - u_col)
                # T3 = A[j, :] @ (u_row - v_row)
                # T4 = A[:, j] @ (u_col - v_col)
                
                # Now, do these sums include k=i and k=j correctly?
                # Let's check T1 for k=i: A[i,i] (B[v, u] - B[u, u]).
                # In the "Terms with i in new" part, we have sum_l A[i,l] B[v, p'[l]].
                # For l=i, p'[i]=v. So A[i,i] B[v, v].
                # In T1, for k=i, we have A[i,i] B[v, p[i]] = A[i,i] B[v, u].
                # This is different.
                
                # The vectorized sums T1..T4 assume p[k] is unchanged for all k.
                # But for k=i, p'[i] = v, not u.
                # For k=j, p'[j] = u, not v.
                
                # So we must subtract the contributions of k=i and k=j from T1..T4 and add the correct new terms.
                
                # T1_corrected = T1 - A[i,i](B[v,u]-B[u,u]) - A[i,j](B[v,v]-B[u,v])
                # Wait, p[i]=u, p[j]=v.
                # T1 term for k=i: A[i,i] (B[v, u] - B[u, u]).
                # T1 term for k=j: A[i,j] (B[v, v] - B[u, v]).
                
                # In the new cost, the term A[i,i] is A[i,i] B[v, v].
                # The term A[i,j] is A[i,j] B[v, u].
                
                # This manual correction is error-prone.
                
                # Given the time limit is generous and n=150, let's just compute the full cost for the candidate swap if it looks promising?
                # No, we need to scan all pairs.
                
                # Let's go back to basics.
                # Cost = sum a[i,j] b[p[i], p[j]].
                # If we swap p[i] and p[j], let u=p[i], v=p[j].
                # New p'[i]=v, p'[j]=u.
                # Delta = sum_{k,l} a[k,l] (b[p'[k], p'[l]] - b[p[k], p[l]])
                # Non-zero only if k or l is i or j.
                
                # Let's just calculate the delta by summing the 4x4 block of interactions involving i and j.
                # The rest of the matrix is unchanged.
                # Old block (rows/cols i,j):
                # [ a[i,i] b[u,u],  a[i,j] b[u,v] ]
                # [ a[j,i] b[v,u],  a[j,i] b[v,v] ] -> No, a[j,i] b[v,u] and a[j,j] b[v,v]
                
                # Old Cost Block = a[i,i]b[u,u] + a[i,j]b[u,v] + a[j,i]b[v,u] + a[j,j]b[v,v]
                # Plus interactions with rest of matrix:
                # sum_{k!=i,j} a[i,k]b[u,p[k]] + a[k,i]b[p[k],u] + a[j,k]b[v,p[k]] + a[k,j]b[p[k],v]
                
                # New Block:
                # [ a[i,i] b[v,v],  a[i,j] b[v,u] ]
                # [ a[j,i] b[u,v],  a[j,j] b[u,u] ]
                
                # New Cost Block = a[i,i]b[v,v] + a[i,j]b[v,u] + a[j,i]b[u,v] + a[j,j]b[u,u]
                # Plus interactions with rest of matrix:
                # sum_{k!=i,j} a[i,k]b[v,p[k]] + a[k,i]b[p[k],v] + a[j,k]b[u,p[k]] + a[k,j]b[p[k],u]
                
                # Delta = (New Block - Old Block) + (New Interactions - Old Interactions)
                
                # This can be computed in O(n) for the interactions.
                
                # Let's implement this.
                
                # Precompute for current p:
                # We need sums like sum_{k!=i,j} a[i,k] b[u, p[k]].
                # We can precompute row sums of b weighted by a?
                
                # Let's just use numpy dot products for the interaction sums.
                # sum_k a[i,k] b[u, p[k]] = a[i, :] @ b[u, p]
                # But this includes k=i and k=j.
                # We must subtract k=i and k=j terms.
                
                b_u_p = b[u, p]
                b_v_p = b[v, p]
                b_p_u = b[p, u]
                b_p_v = b[p, v]
                
                # Sums including all k
                s_i_u = a[i, :] @ b_u_p
                s_i_v = a[i, :] @ b_v_p
                s_j_u = a[j, :] @ b_u_p
                s_j_v = a[j, :] @ b_v_p
                
                s_u_i = a[:, i] @ b_p_u
                s_v_i = a[:, i] @ b_p_v
                s_u_j = a[:, j] @ b_p_u
                s_v_j = a[:, j] @ b_p_v
                
                # Subtract k=i and k=j contributions
                # For s_i_u (sum a[i,k] b[u, p[k]]):
                # k=i: a[i,i] b[u, u]
                # k=j: a[i,j] b[u, v]
                s_i_u_clean = s_i_u - a[i, i]*b[u, u] - a[i, j]*b[u, v]
                
                # For s_i_v (sum a[i,k] b[v, p[k]]):
                # k=i: a[i,i] b[v, u]
                # k=j: a[i,j] b[v, v]
                s_i_v_clean = s_i_v - a[i, i]*b[v, u] - a[i, j]*b[v, v]
                
                # For s_j_u (sum a[j,k] b[u, p[k]]):
                # k=i: a[j,i] b[u, u]
                # k=j: a[j,j] b[u, v]
                s_j_u_clean = s_j_u - a[j, i]*b[u, u] - a[j, j]*b[u, v]
                
                # For s_j_v (sum a[j,k] b[v, p[k]]):
                # k=i: a[j,i] b[v, u]
                # k=j: a[j,j] b[v, v]
                s_j_v_clean = s_j_v - a[j, i]*b[v, u] - a[j, j]*b[v, v]
                
                # For s_u_i (sum a[k,i] b[p[k], u]):
                # k=i: a[i,i] b[u, u]
                # k=j: a[j,i] b[v, u]
                s_u_i_clean = s_u_i - a[i, i]*b[u, u] - a[j, i]*b[v, u]
                
                # For s_v_i (sum a[k,i] b[p[k], v]):
                # k=i: a[i,i] b[u, v]
                # k=j: a[j,i] b[v, v]
                s_v_i_clean = s_v_i - a[i, i]*b[u, v] - a[j, i]*b[v, v]
                
                # For s_u_j (sum a[k,j] b[p[k], u]):
                # k=i: a[i,j] b[u, u]
                # k=j: a[j,j] b[v, u]
                s_u_j_clean = s_u_j - a[i, j]*b[u, u] - a[j, j]*b[v, u]
                
                # For s_v_j (sum a[k,j] b[p[k], v]):
                # k=i: a[i,j] b[u, v]
                # k=j: a[j,j] b[v, v]
                s_v_j_clean = s_v_j - a[i, j]*b[u, v] - a[j, j]*b[v, v]
                
                # Old Interactions = s_i_u_clean + s_u_i_clean + s_j_v_clean + s_v_j_clean
                # New Interactions = s_i_v_clean + s_v_i_clean + s_j_u_clean + s_u_j_clean
                
                # Old Block = a[i,i]b[u,u] + a[i,j]b[u,v] + a[j,i]b[v,u] + a[j,j]b[v,v]
                # New Block = a[i,i]b[v,v] + a[i,j]b[v,u] + a[j,i]b[u,v] + a[j,j]b[u,u]
                
                delta = (s_i_v_clean + s_v_i_clean + s_j_u_clean + s_u_j_clean) - \
                        (s_i_u_clean + s_u_i_clean + s_j_v_clean + s_v_j_clean) + \
                        (a[i,i]*b[v,v] + a[i,j]*b[v,u] + a[j,i]*b[u,v] + a[j,j]*b[u,u]) - \
                        (a[i,i]*b[u,u] + a[i,j]*b[u,v] + a[j,i]*b[v,u] + a[j,j]*b[v,v])
                
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
        
        if best_delta < 0:
            # Perform the swap
            p[best_i], p[best_j] = p[best_j], p[best_i]
            current_cost += best_delta
            improved = True
        else:
            # No improvement found in this pass
            break
            
    return p.tolist()
# EVOLVE-BLOCK-END
