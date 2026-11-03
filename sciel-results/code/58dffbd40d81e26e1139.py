<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    deadline = time.time() + 145.0  # safety margin before 150s kill

    # Symmetrize a (flows) to exploit structure; b (distances) keep as-is
    # a[i,j] * b[p[i],p[j]] — if a is symmetric, cost = sum a[i,j]*b[p[i],p[j]]
    # We'll work with the given matrices directly.

    # Step 1: Get an initial assignment via rank-matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central

    # Compute initial cost
    cost = _compute_cost(a, b, p)

    # Step 2: Hungarian-assisted re-optimization
    # Build a cost matrix for Hungarian: for facility i at location j,
    # approximate marginal cost. Use row-sum matching as a proxy.
    # Actually, let's try a different approach: use the structure of QAP.
    # The cost is sum_{i,j} a[i,j] * b[p[i], p[j]].
    # We can think of this as: for each pair (i,j), the contribution is a[i,j]*b[p[i],p[j]].
    # A good initial solution can come from trying to align high a[i,j] with low b[*,*].

    # Let's try Hungarian on a derived matrix:
    # For facility i, define its "demand vector" d_i[j] = sum_k a[i,k] (row sum)
    # For location j, define its "supply" s_j = sum_k b[j,k] (row sum)
    # But that's just rank matching again.

    # Better: use the fact that cost = trace(A^T P B P^T) where P is permutation matrix.
    # Approximate by minimizing sum_i (A @ P B P^T)[i,i] = sum_i sum_j a[i,j] b[p[i],p[j]]
    # which is exactly our cost.

    # Let's try a simple local search with 2-opt swaps.
    # For swapping p[i] and p[j], the delta in cost can be computed efficiently.

    # Precompute for fast delta computation:
    # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # If we swap p[i] <-> p[j], only terms involving i or j change.
    # Old terms involving i: sum_j a[i,j]*b[p[i],p[j]] + sum_j a[j,i]*b[p[j],p[i]]
    # New terms: sum_j a[i,j]*b[p[j],p[j]] ... wait, need to be careful.

    # Let's define:
    # Let x = p[i], y = p[j] (the two locations being swapped)
    # Old contribution from pairs (i,*) and (*,i) and (j,*) and (*,j):
    #   sum_k a[i,k]*b[x, p[k]] + sum_k a[k,i]*b[p[k], x] + sum_k a[j,k]*b[y, p[k]] + sum_k a[k,j]*b[p[k], y]
    #   minus a[i,j]*b[x,y] and a[j,i]*b[y,x] which are counted twice? No, let's think again.

    # Actually, let's just compute the full delta by iterating over all k != i, j:
    # delta = sum_{k != i, j} [a[i,k]*b[y,p[k]] + a[k,i]*b[p[k],y] + a[j,k]*b[x,p[k]] + a[k,j]*b[p[k],x]
    #          - a[i,k]*b[x,p[k]] - a[k,i]*b[p[k],x] - a[j,k]*b[y,p[k]] - a[k,j]*b[p[k],y]]
    #         + a[i,j]*b[y,x] + a[j,i]*b[x,y] - a[i,j]*b[x,y] - a[j,i]*b[y,x]

    # This is O(n) per swap, and we try O(n^2) swaps per pass, so O(n^3) per pass.
    # For n=150, that's ~3.4M operations per pass, which is fast.

    # Let's implement a simple 2-opt local search.

    # First, let's try to improve the initial solution with a few passes of 2-opt.
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        # Precompute for efficiency
        for i in range(n):
            if time.time() > deadline - 5.0:
                break
            for j in range(i + 1, n):
                # Compute delta for swapping p[i] and p[j]
                x, y = p[i], p[j]
                delta = 0
                for k in range(n):
                    if k == i or k == j:
                        continue
                    pk = p[k]
                    delta += a[i, k] * (b[y, pk] - b[x, pk])
                    delta += a[k, i] * (b[pk, y] - b[pk, x])
                    delta += a[j, k] * (b[x, pk] - b[y, pk])
                    delta += a[k, j] * (b[pk, x] - b[pk, y])
                # Terms involving i and j directly
                delta += a[i, j] * (b[y, x] - b[x, y])
                delta += a[j, i] * (b[x, y] - b[y, x])

                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    cost += delta
                    improved = True
        if time.time() > deadline - 5.0:
            break

    return p.tolist()


def _compute_cost(a, b, p):
    n = len(a)
    cost = 0
    for i in range(n):
        for j in range(n):
            cost += a[i, j] * b[p[i], p[j]]
    return cost
# EVOLVE-BLOCK-END
=======
"""QAP solver for tho150 using vectorized local search with ILS."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def solve(a, b):
    n = len(a)
    deadline = time.time() + 145.0  # safety margin before 150s kill

    # Convert to float64 for faster arithmetic
    a = a.astype(np.float64)
    b = b.astype(np.float64)

    best_p = None
    best_cost = np.inf

    def compute_cost(p):
        """Compute QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]"""
        return np.sum(a * b[np.ix_(p, p)])

    def local_search(p, cost):
        """Steepest-descent 2-opt local search with vectorized delta computation."""
        n = len(p)
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            best_delta = 0
            best_i = -1
            best_j = -1

            for i in range(n):
                if time.time() > deadline - 5.0:
                    break

                x = p[i]
                # For each other facility j, compute delta of swapping p[i] and p[j]
                # Vectorize over j
                # delta_ij = sum_{k != i,j} a[i,k]*(b[p[j],p[k]] - b[x,p[k]])
                #          + sum_{k != i,j} a[k,i]*(b[p[k],p[j]] - b[p[k],x])
                #          + sum_{k != i,j} a[j,k]*(b[x,p[k]] - b[p[j],p[k]])
                #          + sum_{k != i,j} a[k,j]*(b[p[k],x] - b[p[k],p[j]])
                #          + a[i,j]*(b[p[j],x] - b[x,p[j]])
                #          + a[j,i]*(b[x,p[j]] - b[p[j],x])

                # Precompute vectors for facility i
                ai = a[i]  # a[i, :]
                aiT = a[:, i]  # a[:, i]

                # For all j != i, we want to compute the delta
                # Let's compute terms that don't depend on j first:
                # Term 1: sum_{k != i} a[i,k] * (b[y, p[k]] - b[x, p[k]]) where y = p[j]
                # = sum_k a[i,k]*b[y,p[k]] - sum_k a[i,k]*b[x,p[k]]
                # The second part is constant for fixed i: c1 = sum_k a[i,k]*b[x,p[k]]
                c1 = np.dot(ai, b[x, p])

                # Term 2: sum_{k != i} a[k,i] * (b[p[k],y] - b[p[k],x])
                # = sum_k a[k,i]*b[p[k],y] - sum_k a[k,i]*b[p[k],x]
                # Second part constant: c2 = sum_k a[k,i]*b[p[k],x]
                c2 = np.dot(aiT, b[p, x])

                # Now for each j, we need:
                # sum_k a[i,k]*b[p[j],p[k]] - c1
                # + sum_k a[k,i]*b[p[k],p[j]] - c2
                # + terms involving j directly

                # Let's compute all deltas for j != i at once
                # For each j, y = p[j]
                # term1_j = sum_k a[i,k]*b[y,p[k]] = b[y, p] @ ai  (dot product)
                # term2_j = sum_k a[k,i]*b[p[k],y] = aiT @ b[p, y]

                # Vectorize: for all j, compute b[p[j], p] @ ai
                # b[p[j], p] is a row of b indexed by p[j], columns p
                # This is b[np.ix_(p, p)] @ ai ... no wait.
                # b[p[j], p[k]] for all k = b[p[j], :] indexed by p
                # Let B_pp = b[np.ix_(p, p)]  # B_pp[i,j] = b[p[i], p[j]]
                # Then term1_j = B_pp[j, :] @ ai = np.dot(B_pp[j], ai)
                # term2_j = B_pp[:, j] @ aiT = np.dot(B_pp[:, j], aiT)

                B_pp = b[np.ix_(p, p)]  # n x n matrix, B_pp[i,j] = b[p[i], p[j]]

                # term1 for all j: B_pp @ ai  (n-vector)
                term1_all = B_pp @ ai  # term1_all[j] = sum_k B_pp[j,k]*ai[k] = sum_k b[p[j],p[k]]*a[i,k]
                # term2 for all j: B_pp.T @ aiT (n-vector)
                term2_all = B_pp.T @ aiT  # term2_all[j] = sum_k B_pp[k,j]*aiT[k] = sum_k b[p[k],p[j]]*a[k,i]

                # Now the j-dependent terms from the "other facility j" part:
                # sum_k a[j,k]*(b[x,p[k]] - b[p[j],p[k]])
                # = sum_k a[j,k]*b[x,p[k]] - sum_k a[j,k]*b[p[j],p[k]]
                # First part: a[j,:] @ b[x, p]
                # Second part: a[j,:] @ b[p[j], p] = a[j,:] @ B_pp[j,:]

                # sum_k a[k,j]*(b[p[k],x] - b[p[k],p[j]])
                # = sum_k a[k,j]*b[p[k],x] - sum_k a[k,j]*b[p[k],p[j]]
                # First part: a[:,j] @ b[p, x]
                # Second part: a[:,j] @ B_pp[:,j]

                # Direct terms:
                # a[i,j]*(b[p[j],x] - b[x,p[j]]) + a[j,i]*(b[x,p[j]] - b[p[j],x])
                # = (a[i,j] - a[j,i])*(b[p[j],x] - b[x,p[j]])
                # Note: b[p[j],x] = B_pp[j, i] and b[x,p[j]] = B_pp[i, j]

                # Let's compute for all j:
                # part3_j = a[j,:] @ b[x, p] - a[j,:] @ B_pp[j,:]
                # part4_j = a[:,j] @ b[p, x] - a[:,j] @ B_pp[:,j]
                # part5_j = (a[i,j] - a[j,i]) * (B_pp[j, i] - B_pp[i, j])

                # part3: for all j, a[j,:] @ b[x, p] is a row of a @ b[x, p] ... 
                # a @ b[x, p] gives a vector where element j = sum_k a[j,k]*b[x,p[k]]
                part3_first = a @ b[x, p]  # part3_first[j] = sum_k a[j,k]*b[x,p[k]]

                # a[j,:] @ B_pp[j,:] for all j: this is diagonal of (a * B_pp)
                # (a * B_pp)[j,j] = a[j,j]*B_pp[j,j] ... no, we need sum_k a[j,k]*B_pp[j,k]
                # That's row-wise dot product of a and B_pp
                part3_second = np.einsum('jk,jk->j', a, B_pp)

                # part4: a[:,j] @ b[p, x] for all j
                # = (b[p, x] @ a) ... element j = sum_k b[p[k],x]*a[k,j] = (b[p,x] @ a)[j]
                part4_first = b[p, x] @ a  # part4_first[j] = sum_k b[p[k],x]*a[k,j]

                # a[:,j] @ B_pp[:,j] for all j: column-wise dot product
                part4_second = np.einsum('jk,jk->j', a, B_pp)

                # part5: (a[i,:] - a[:,i]) * (B_pp[:, i] - B_pp[i, :])
                # Note: a[i,j] - a[j,i] = (a[i,:] - a[:,i])[j]
                # B_pp[j,i] - B_pp[i,j] = (B_pp[:,i] - B_pp[i,:])[j]
                part5 = (a[i, :] - a[:, i]) * (B_pp[:, i] - B_pp[i, :])

                # Total delta for swapping i and j:
                # delta_j = (term1_all[j] - c1) + (term2_all[j] - c2)
                #         + (part3_first[j] - part3_second[j])
                #         + (part4_first[j] - part4_second[j])
                #         + part5[j]
                # But we need to exclude k=i and k=j from the sums.
                # The above sums include k=i and k=j. Let's correct:
                # For term1: sum_{k != i,j} a[i,k]*b[y,p[k]]
                #   = sum_{k != i} a[i,k]*b[y,p[k]] - a[i,j]*b[y,p[j]]
                #   = term1_all[j] - a[i,j]*b[p[j],p[j]]
                #   But also need to subtract k=i: a[i,i]*b[y,p[i]] = a[i,i]*b[p[j],x]
                #   So: term1_corrected = term1_all[j] - a[i,j]*B_pp[j,j] - a[i,i]*B_pp[j,i]
                
                # This is getting complicated. Let me just compute delta directly but vectorized per i.
                
                # Actually, let's use a simpler but still vectorized approach:
                # Compute delta for all j at once using the full formula, then mask out j=i.
                
                # Full delta (without excluding i,j from sums):
                # delta_j = term1_all[j] - c1 + term2_all[j] - c2
                #         + part3_first[j] - part3_second[j]
                #         + part4_first[j] - part4_second[j]
                #         + part5[j]
                
                deltas = (term1_all - c1) + (term2_all - c2) \
                        + (part3_first - part3_second) \
                        + (part4_first - part4_second) \
                        + part5

                # Now correct for the over-counting of k=i and k=j terms.
                # The sums in term1, term2, part3, part4 include k=i and k=j.
                # We need to subtract the contributions from k=i and k=j.
                
                # For k=i:
                # term1 includes a[i,i]*b[y,p[i]] = a[i,i]*b[p[j],x] = a[i,i]*B_pp[j,i]
                # term2 includes a[i,i]*b[p[i],y] = a[i,i]*b[x,p[j]] = a[i,i]*B_pp[i,j]
                # part3 includes a[j,i]*b[x,p[i]] - a[j,i]*b[p[j],p[i]] = a[j,i]*(b[x,x] - b[p[j],x]) = a[j,i]*(b[x,x] - B_pp[j,i])
                #   Wait, this is getting messy. Let me just accept a small error and rely on the fact
                # that for large n, the correction terms are small relative to the total.
                # Alternatively, compute delta more carefully.
                
                # Let me just compute the delta for each j individually but in a semi-vectorized way.
                # For now, let's find the best j for this i by scanning.
                
                # Mask out j=i
                deltas[i] = np.inf  # Can't swap i with itself
                
                # Find best j for this i
                j_best = np.argmin(deltas)
                d_best = deltas[j_best]
                
                if d_best < best_delta:
                    best_delta = d_best
                    best_i = i
                    best_j = j_best

            if best_i >= 0 and best_delta < 0:
                p[best_i], p[best_j] = p[best_j], p[best_i]
                cost += best_delta
                improved = True

        return p, cost

    # Multiple restarts with different initial solutions
    n_restarts = 0
    while time.time() < deadline - 10.0:
        n_restarts += 1

        if n_restarts == 1:
            # Initial: rank matching
            heavy = np.argsort(-a.sum(axis=1), kind="stable")
            central = np.argsort(b.sum(axis=1), kind="stable")
            p = np.empty(n, dtype=np.int64)
            p[heavy] = central
        else:
            # Perturbed version of best solution
            if best_p is not None:
                p = best_p.copy()
                # Randomly swap 10-20 pairs
                n_swaps = random.randint(10, 20)
                for _ in range(n_swaps):
                    i, j = random.sample(range(n), 2)
                    p[i], p[j] = p[j], p[i]
            else:
                p = np.random.permutation(n).astype(np.int64)

        cost = compute_cost(p)
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()

        # Local search
        p, cost = local_search(p, cost)

        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()

        # Check time
        if time.time() > deadline - 10.0:
            break

    if best_p is None:
        # Fallback: random permutation
        best_p = np.random.permutation(n).astype(np.int64)

    return best_p.tolist()


def _compute_cost(a, b, p):
    n = len(a)
    cost = 0
    for i in range(n):
        for j in range(n):
            cost += a[i, j] * b[p[i], p[j]]
    return cost
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, I realize the vectorized delta computation has issues with the over-counting corrections. Let me simplify by using a cleaner vectorized approach that computes the delta correctly.

