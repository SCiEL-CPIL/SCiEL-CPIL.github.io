"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def _precompute(max_n):
    """
    Precompute LCM table and bad triples.
    Returns:
    lcm_table: 2D list
    bad_triples: List of tuples (a, b, c)
    triple_sets: List of sets for each triple (for fast membership)
    element_to_triples: Dict mapping element -> list of indices of bad triples it's in
    """
    n = max_n
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            if i != j:
                lcm_table[j][i] = lcm_table[i][j]
    
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_triples.append((a, b, c))
    
    # Map element to list of triple indices for fast conflict checking
    element_to_triples = {i: [] for i in range(1, n + 1)}
    for idx, (a, b, c) in enumerate(bad_triples):
        element_to_triples[a].append(idx)
        element_to_triples[b].append(idx)
        element_to_triples[c].append(idx)
        
    return lcm_table, bad_triples, element_to_triples

def _is_valid_set(S, bad_triples, element_to_triples):
    """Check if S contains any bad triple."""
    S_set = set(S)
    for a, b, c in bad_triples:
        if a in S_set and b in S_set and c in S_set:
            return False
    return True

def _greedy_construct(order, bad_triples, element_to_triples, max_n):
    """
    Greedily add elements from order to S if they don't form a bad triple with existing elements.
    """
    S = []
    S_set = set()
    # For fast checking, we can just check if adding x creates a bad triple.
    # A bad triple is formed if there exist a, b in S such that (x, a, b) is a bad triple.
    # We can precompute for each x, the list of pairs (a,b) such that (x,a,b) is bad.
    # But checking against S is O(|S| * degree(x)).
    
    # Optimization: Precompute for each x, the set of "conflict pairs" or just check directly.
    # Since N=128, direct check is fast enough.
    
    for x in order:
        if x in S_set:
            continue
        # Check if x conflicts with any pair in S
        conflict = False
        for i in range(len(S)):
            a = S[i]
            for j in range(i + 1, len(S)):
                b = S[j]
                # Check if (x, a, b) is a bad triple
                # We need a fast way to check this.
                # Precomputed: bad_triples_set? No, too many.
                # Let's use the element_to_triples mapping? 
                # If we have a set of active triples, we can check if any triple containing x is fully in S.
                # But S is growing.
                
                # Simpler: Just check LCMs.
                # lcm(x,a) == lcm(x,b) == lcm(a,b)
                # We have lcm_table.
                if _lcm(x, a) == _lcm(x, b) == _lcm(a, b):
                    conflict = True
                    break
            if conflict:
                break
        
        if not conflict:
            S.append(x)
            S_set.add(x)
            
    return S

def _local_search(S, bad_triples, element_to_triples, max_n, deadline):
    """
    Improve S by adding elements or swapping.
    """
    S_set = set(S)
    S_list = list(S)
    full_set = set(range(1, max_n + 1))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Phase 1: Try to add any element not in S
        for x in full_set - S_set:
            # Check if adding x creates a bad triple
            conflict = False
            for i in range(len(S_list)):
                a = S_list[i]
                for j in range(i + 1, len(S_list)):
                    b = S_list[j]
                    if _lcm(x, a) == _lcm(x, b) == _lcm(a, b):
                        conflict = True
                        break
                if conflict:
                    break
            
            if not conflict:
                S_list.append(x)
                S_set.add(x)
                improved = True
                break
        
        if improved:
            continue
        
        # Phase 2: Try 1-swaps (remove one, add one)
        # We want to maintain or increase size. Since we failed to add, size is fixed.
        # A swap keeps size same. We might do this to reach a better state for future adds?
        # Or just to find a larger set if we get stuck?
        # Actually, if we can't add, a swap doesn't help increase size immediately.
        # But it might allow a subsequent add.
        # Let's try swaps that reduce the "conflict potential" or just random swaps to escape local minima.
        
        # Better: Try to remove an element that is "high conflict" and see if we can add two? No, that's 2-swap.
        # Let's stick to 1-swap for now, but only accept if it allows an immediate add? 
        # Or just cycle through swaps to find a configuration where an add is possible.
        
        found_swap = False
        # Sort S_list by "conflict score" (number of bad triples it forms with other elements in S)
        # This is a heuristic.
        
        for a in S_list:
            S_set.remove(a)
            S_list.remove(a)
            
            # Try to add any x not in S_set
            for x in full_set - S_set:
                conflict = False
                for i in range(len(S_list)):
                    u = S_list[i]
                    for j in range(i + 1, len(S_list)):
                        v = S_list[j]
                        if _lcm(x, u) == _lcm(x, v) == _lcm(u, v):
                            conflict = True
                            break
                    if conflict:
                        break
                
                if not conflict:
                    # Accept swap: remove a, add x
                    S_list.append(x)
                    S_set.add(x)
                    found_swap = True
                    improved = True
                    break
            
            if found_swap:
                break
            
            # Restore a
            S_set.add(a)
            S_list.append(a)
            
            if time.time() > deadline:
                break
        
        if not found_swap:
            # Try 2-swaps? Or break.
            # Given time constraints, 1-swap is often sufficient for N=128 if greedy is good.
            break
            
    return S_list

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s limit to be safe
    
    max_n = 128
    lcm_table, bad_triples, element_to_triples = _precompute(max_n)
    
    if not bad_triples:
        return list(range(1, max_n + 1))
        
    best_set = []
    best_size = 0
    
    # Define multiple orderings
    all_elements = list(range(1, max_n + 1))
    
    # 1. Natural order
    orderings = [all_elements]
    
    # 2. Reverse order
    orderings.append(list(reversed(all_elements)))
    
    # 3. Order by number of bad triples (ascending)
    def count_conflicts(x):
        count = 0
        for idx in element_to_triples[x]:
            a, b, c = bad_triples[idx]
            # Count how many of the other two are "likely" to be in a set?
            # Just count total triples for now.
            count += 1
        return count
        
    # Actually, a better heuristic is to sort by the number of *potential* conflicts.
    # Let's just use the degree in the hypergraph.
    degrees = {x: len(element_to_triples[x]) for x in all_elements}
    orderings.append(sorted(all_elements, key=lambda x: degrees[x]))
    orderings.append(sorted(all_elements, key=lambda x: -degrees[x]))
    
    # 4. Random orderings
    random.seed(42)
    for _ in range(50):
        orderings.append(random.sample(all_elements, max_n))
        
    # Run Greedy
    for order in orderings:
        if time.time() > deadline - 10:
            break
        S = _greedy_construct(order, bad_triples, element_to_triples, max_n)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            
    # Local Search on the best set
    if time.time() < deadline - 5:
        improved_set = _local_search(best_set, bad_triples, element_to_triples, max_n, deadline)
        if len(improved_set) > best_size:
            best_set = improved_set
            best_size = len(best_set)
            
    # Final Validation
    S_set = set(best_set)
    valid = True
    for a, b, c in bad_triples:
        if a in S_set and b in S_set and c in S_set:
            valid = False
            break
            
    if not valid:
        # Fallback to a known valid set (e.g., all primes, or just {1})
        # Primes are safe because lcm(p, q) = pq for distinct primes, and lcm(p, p) is not distinct.
        # If we pick only primes, any triple of distinct primes p,q,r has lcm(p,q)=pq, lcm(p,r)=pr, lcm(q,r)=qr.
        # pq = pr = qr implies p=q=r, contradiction.
        # So all primes is a valid set.
        primes = [x for x in range(2, max_n + 1) if all(x % i != 0 for i in range(2, int(x**0.5) + 1))]
        if len(primes) > len(best_set):
            best_set = primes
            
    if not best_set:
        best_set = [1]
        
    return best_set

# EVOLVE-BLOCK-END
