"""Program for integer sum-product extremal sets."""
import itertools
import random
from collections import Counter

# EVOLVE-BLOCK-START
def construct():
    """
    Construct 20 distinct positive integers <= 10000 minimizing max(|A+A|, |A*A|).
    
    Strategy:
    1. Start with a geometric progression A = {b^(e_i)} which has small product set.
    2. Perturb with small multipliers k_i to reduce sum set size while keeping product set manageable.
    3. Use local search to refine the set.
    """
    
    def compute_sets(A):
        """Compute sum set and product set sizes."""
        A = sorted(A)
        n = len(A)
        sum_set = set()
        prod_set = set()
        for i in range(n):
            for j in range(i, n):
                sum_set.add(A[i] + A[j])
                prod_set.add(A[i] * A[j])
        return len(sum_set), len(prod_set)
    
    def score(A):
        s, p = compute_sets(A)
        return max(s, p)
    
    # Step 1: Generate candidate geometric progressions
    # A = {b^(e_i)} where e_i are exponents
    # Product set of {b^(e_i)} is {b^(e_i + e_j)} for i <= j
    # So |A*A| = number of distinct sums e_i + e_j for i <= j
    # Sum set is {b^(e_i) + b^(e_j)} which can be large
    
    # Let's try small bases and exponent sets
    best_A = None
    best_score = float('inf')
    
    # Try base 2 with various exponent patterns
    # Exponents should be chosen so that e_i + e_j are distinct (to keep prod set small)
    # but also so that b^(e_i) + b^(e_j) collide as much as possible (to keep sum set small)
    
    # For base 2, 2^a + 2^b = 2^c + 2^d iff {a,b} = {c,d}
    # So sum set size = n*(n+1)/2 for geometric progression with base 2
    # That's 210 for n=20, which is bad.
    
    # For product set: 2^(a) * 2^(b) = 2^(a+b), so |A*A| = number of distinct a+b for a,b in E, a<=b
    # If E = {0, 1, 2, ..., 19}, then a+b ranges from 0 to 38, so 39 distinct values.
    # So max(210, 39) = 210.
    
    # We need to do better on the sum set.
    
    # Alternative: use a set where elements are close together so sums collide.
    # But then products will be very different.
    
    # Let's try a hybrid: start with a geometric progression and perturb.
    
    # Try base 2, exponents 0..19
    base = 2
    exponents = list(range(20))
    A = [base ** e for e in exponents if base ** e <= 10000]
    # 2^0=1, 2^1=2, ..., 2^13=8192, 2^14=16384 > 10000
    # So only 14 elements. Need to extend.
    
    # Let's try a different approach: use a set of numbers that are products of small primes.
    # Or use a structured set.
    
    # Approach: Use local search starting from a reasonable initial set.
    
    # Initial set: try a geometric progression with base 2, but cap at 10000
    # and fill remaining with small perturbations.
    
    # Let's try: A = {2^i for i in 0..13} union {some other small numbers}
    # But they need to be distinct.
    
    # Better initial guess: use a set of smooth numbers.
    # Or start with range(1, 21) and do local search.
    
    # Let's use simulated annealing / hill climbing.
    
    random.seed(42)
    
    # Initial candidate: geometric progression with base 2
    A_init = [2**i for i in range(14)]  # 1 to 8192
    # Need 6 more numbers <= 10000, distinct from above
    # Add: 3, 5, 7, 9, 11, 13 (but 3,5,7,9,11,13 might conflict with powers of 2? No, powers of 2 are even except 1)
    # Actually 1 is in the set. Let's add odd numbers not in the set.
    existing = set(A_init)
    extra = []
    for x in range(3, 10001, 2):
        if x not in existing and len(extra) < 6:
            extra.append(x)
        if len(extra) >= 6:
            break
    A = A_init + extra
    A = list(set(A))
    if len(A) != 20:
        # Fill with more numbers
        for x in range(3, 10001):
            if x not in A and len(A) < 20:
                A.append(x)
            if len(A) >= 20:
                break
    
    A = sorted(A[:20])
    current_score = score(A)
    best_A = list(A)
    best_score = current_score
    
    # Local search: try replacing one element with another
    # or swapping two elements
    
    # Precompute for speed
    def score_fast(A):
        n = len(A)
        sum_set = set()
        prod_set = set()
        for i in range(n):
            ai = A[i]
            for j in range(i, n):
                aj = A[j]
                sum_set.add(ai + aj)
                prod_set.add(ai * aj)
        return max(len(sum_set), len(prod_set))
    
    # Hill climbing with random restarts
    for restart in range(5):
        if restart == 0:
            A = list(best_A)
        else:
            # Random perturbation of best_A
            A = list(best_A)
            for _ in range(5):
                idx = random.randint(0, 19)
                new_val = random.randint(1, 10000)
                if new_val not in A:
                    A[idx] = new_val
            A = sorted(set(A))
            # Ensure 20 distinct
            while len(A) < 20:
                A.append(random.randint(1, 10000))
            A = sorted(list(set(A))[:20])
        
        current_score = score_fast(A)
        
        # Try swaps and replacements
        improved = True
        while improved:
            improved = False
            # Try replacing each element
            for i in range(20):
                # Try a few random replacements
                for _ in range(10):
                    new_val = random.randint(1, 10000)
                    if new_val in A:
                        continue
                    A_new = list(A)
                    A_new[i] = new_val
                    A_new = sorted(set(A_new))
                    if len(A_new) != 20:
                        continue
                    s_new = score_fast(A_new)
                    if s_new < current_score:
                        A = A_new
                        current_score = s_new
                        if current_score < best_score:
                            best_score = current_score
                            best_A = list(A)
                        improved = True
                        break
            # Try swapping two elements
            for i in range(20):
                for j in range(i+1, 20):
                    A_new = list(A)
                    A_new[i], A_new[j] = A_new[j], A_new[i]
                    A_new = sorted(set(A_new))
                    if len(A_new) != 20:
                        continue
                    s_new = score_fast(A_new)
                    if s_new < current_score:
                        A = A_new
                        current_score = s_new
                        if current_score < best_score:
                            best_score = current_score
                            best_A = list(A)
                        improved = True
    
    # Final: try a structured approach with small primes
    # Use products of small primes to get many collisions in both sum and product sets
    
    # Try: A = {p_i * q_i} where p_i, q_i are small primes
    # This gives structure to both sums and products
    
    # Let's try a few structured candidates
    structured_candidates = []
    
    # Candidate 1: Powers of 2 up to 2^13, plus 3*powers of 2
    c1 = [2**i for i in range(14)]
    c1 += [3 * 2**i for i in range(6) if 3 * 2**i not in set(c1) and 3 * 2**i <= 10000]
    c1 = sorted(set(c1)[:20])
    if len(c1) == 20:
        structured_candidates.append(c1)
    
    # Candidate 2: Powers of 3
    c2 = [3**i for i in range(10) if 3**i <= 10000]  # 3^0=1 to 3^8=6561, 3^9=19683
    c2 += [2 * 3**i for i in range(10) if 2 * 3**i <= 10000 and 2 * 3**i not in set(c2)]
    c2 = sorted(set(c2)[:20])
    if len(c2) == 20:
        structured_candidates.append(c2)
    
    # Candidate 3: Mixed: 2^a * 3^b
    c3 = []
    for a in range(14):
        for b in range(7):
            val = (2**a) * (3**b)
            if val <= 10000 and val not in c3:
                c3.append(val)
            if len(c3) >= 20:
                break
        if len(c3) >= 20:
            break
    c3 = sorted(c3[:20])
    if len(c3) == 20:
        structured_candidates.append(c3)
    
    # Evaluate structured candidates
    for cand in structured_candidates:
        s = score_fast(cand)
        if s < best_score:
            best_score = s
            best_A = list(cand)
    
    # Final local search on best_A
    A = list(best_A)
    current_score = best_score
    
    for _ in range(100):
        improved = False
        for i in range(20):
            for _ in range(20):
                new_val = random.randint(1, 10000)
                if new_val in A:
                    continue
                A_new = list(A)
                A_new[i] = new_val
                A_new = sorted(set(A_new))
                if len(A_new) != 20:
                    continue
                s_new = score_fast(A_new)
                if s_new < current_score:
                    A = A_new
                    current_score = s_new
                    best_score = current_score
                    best_A = list(A)
                    improved = True
                    break
            if improved:
                break
    
    return best_A
# EVOLVE-BLOCK-END
