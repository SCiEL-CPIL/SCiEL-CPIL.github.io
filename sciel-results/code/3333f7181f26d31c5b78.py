"""
Strategy: Backtracking Search with Forward Checking and MRV Heuristic.
Instead of local search, we model this as a Constraint Satisfaction Problem (CSP).
We try to construct a set of size k by selecting elements from 2..40.
We use backtracking to explore combinations.
To make it fast:
1. We pre-compute factorizations.
2. We use a 'seen' set of exponent vectors for the current subset.
3. We use MRV (Minimum Remaining Values) to choose the next candidate number to add.
   Actually, since we are adding numbers, the 'domain' is the set of unused numbers.
   The constraint is that the new number's factor vector, added to any existing subset vector,
   must not collide with any existing subset vector.
   Wait, the condition is: for all A, B in subsets, prod(A) != prod(B).
   If we have a current set S with distinct subset products, and we want to add x,
   we need to ensure that for any subset T of S, prod(T) * x != prod(U) for any subset U of S.
   Equivalently, x * v_T != v_U for all T, U.
   Since v_U are all distinct, this is equivalent to:
   For any T, U, if v_T * x == v_U, then collision.
   This is hard to check directly without enumerating.
   
   Better approach for Backtracking:
   We maintain the set of all current subset product vectors (as tuples of exponents).
   Let S be the current set of chosen numbers.
   Let V_S be the set of all 2^|S| exponent vectors.
   To add a candidate x with factor vector f_x:
   The new set of vectors will be V_S ∪ {v + f_x for v in V_S}.
   We need V_S ∩ {v + f_x for v in V_S} to be empty? No.
   We need all elements in the new set to be distinct.
   Since elements in V_S are distinct, and elements in {v+f_x} are distinct (shifted),
   we only need to check if any v1 + f_x == v2 for v1, v2 in V_S.
   i.e., v2 - v1 == f_x.
   So, for a candidate x to be valid, its factor vector f_x must NOT be equal to the difference
   of any two distinct vectors in V_S.
   
   This is a very efficient check!
   Pre-compute all differences v2 - v1 for v1, v2 in V_S?
   That would be 2^|S| * 2^|S| differences, which is 4^|S|.
   For |S|=10, 4^10 = 1M. For |S|=15, 4^15 = 1B. Too slow.
   
   Alternative:
   Check on the fly.
   For candidate x with f_x:
   For each v in V_S:
       target = v + f_x
       if target in V_S:
           collision!
   
   This is O(2^|S|) per candidate.
   Number of candidates ~ 40.
   So O(40 * 2^|S|) per step.
   For |S|=15, 40 * 32768 = 1.3M operations.
   Depth 15. Total operations ~ 15 * 1.3M = 20M. This is feasible in Python if optimized.
   
   To improve further, we can use MRV?
   Actually, the order of adding numbers matters for the search tree size.
   If we add numbers that "constrain" the future more, we prune earlier.
   Numbers with large prime factors or unique structures might be good.
   However, standard backtracking with the O(2^|S|) check is a solid structural change from local search.
   
   Let's implement Backtracking with:
   1. Pre-computed factor vectors.
   2. Dynamic set of current subset vectors.
   3. Check: for v in current_vectors: if v + f_x in current_vectors: fail.
   4. To speed up, we can order candidates.
   5. We will try to find a set of size 18. If not found, 17, etc.
"""

import time
import sys

# Configuration
MAX_VAL = 40
MAX_SIZE = 18
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
NUM_PRIMES = len(PRIMES)

# Precompute factorization for 2..40
def _factorize(n):
    exponents = [0] * NUM_PRIMES
    for i, p in enumerate(PRIMES):
        while n % p == 0:
            exponents[i] += 1
            n //= p
    return tuple(exponents)

FACTORS = {i: _factorize(i) for i in range(2, MAX_VAL + 1)}

# Global deadline
_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0  # Safety margin

def _time_exceeded():
    return time.time() > _DEADLINE

class Backtracker:
    def __init__(self):
        self.current_set = []
        self.current_vectors = set()
        self.current_vectors.add((0,) * NUM_PRIMES)
        self.best_size = 0
        self.best_set = []
        
    def _add_candidate(self, x):
        """
        Attempt to add candidate x to the current set.
        Returns True if successful, False otherwise.
        """
        f_x = FACTORS[x]
        
        # Check for collision:
        # New vectors are {v + f_x for v in current_vectors}
        # We need to ensure no (v + f_x) is in current_vectors.
        # Also, we assume v1 + f_x != v2 + f_x if v1 != v2, which is true.
        # And v1 + f_x != v2 (for v1, v2 in current_vectors) is the condition.
        
        # Optimization:
        # Instead of iterating all v, we can just check.
        # For v in current_vectors:
        #    if v + f_x in self.current_vectors: return False
        
        # This is O(N) where N = 2^k.
        
        new_vecs = set()
        for v in self.current_vectors:
            nv = tuple(v[i] + f_x[i] for i in range(NUM_PRIMES))
            if nv in self.current_vectors:
                return False
            new_vecs.add(nv)
            
        # If we get here, it's valid
        self.current_set.append(x)
        self.current_vectors.update(new_vecs)
        return True

    def _remove_candidate(self, x):
        """
        Remove the last added candidate x.
        """
        self.current_set.pop()
        f_x = FACTORS[x]
        
        # Remove vectors that were added by x.
        # These are v + f_x for v in the previous set of vectors.
        # But we don't have the previous set stored explicitly.
        # We can reconstruct it:
        # Current vectors = V_old ∪ (V_old + f_x)
        # V_old = {v for v in current_vectors if v - f_x is not in current_vectors? No.}
        
        # Simpler: Just recompute the set of vectors from self.current_set.
        # This is O(2^k * k) which is acceptable for k <= 18.
        self._recompute_vectors()

    def _recompute_vectors(self):
        """Recompute the set of all subset product vectors from current_set."""
        if not self.current_set:
            self.current_vectors = {(0,) * NUM_PRIMES}
            return
        
        # Start with empty product
        vectors = set()
        vectors.add((0,) * NUM_PRIMES)
        
        for x in self.current_set:
            f_x = FACTORS[x]
            # Add x to all existing subsets
            new_vectors = set()
            for v in vectors:
                nv = tuple(v[i] + f_x[i] for i in range(NUM_PRIMES))
                new_vectors.add(nv)
            vectors.update(new_vectors)
            
        self.current_vectors = vectors

    def _search(self, target_size, candidates):
        """
        Recursively search for a set of size target_size.
        candidates: list of available numbers.
        Returns True if found, False otherwise.
        """
        if _time_exceeded():
            return False
            
        if len(self.current_set) == target_size:
            if target_size > self.best_size:
                self.best_size = target_size
                self.best_set = self.current_set[:]
            return True
            
        # Heuristic: Try candidates in a specific order.
        # Maybe try numbers with larger prime factors first?
        # Or just iterate through candidates.
        
        # To avoid redundant work, we enforce an order on the set.
        # We only pick candidates > last picked?
        # No, the set is unordered, but to avoid permutations,
        # we can enforce that we pick from candidates in increasing order?
        # If we do that, we miss sets where a smaller number is added later.
        # Standard combination generation:
        # We maintain a list of available candidates.
        # We pick one, recurse, then un-pick.
        # To ensure we don't pick the same combination in different orders,
        # we can just iterate through the list of candidates and remove the picked one
        # from the list for the recursive call.
        
        for i in range(len(candidates)):
            if _time_exceeded():
                return False
                
            x = candidates[i]
            
            # Try adding x
            if self._add_candidate(x):
                # Recurse with remaining candidates
                # To avoid permutations, we only consider candidates AFTER i?
                # No, that restricts the set to be sorted.
                # If we restrict to sorted order, we cover all unique sets exactly once.
                remaining = candidates[i+1:]
                if self._search(target_size, remaining):
                    return True
                
                # Undo
                self._remove_candidate(x)
                
        return False

def construct():
    """
    Construct a set of distinct integers in 2..40 with distinct subset products.
    Maximize the number of selected integers.
    """
    start_time = time.time()
    
    # We want to maximize size.
    # Max possible is 18.
    # We try sizes from 18 down to 1.
    
    # To make backtracking faster, we can use a heuristic ordering of candidates.
    # Candidates that are "more independent" (e.g., primes) are better to place early?
    # Or maybe we should just try all.
    
    # Let's try to find a set of size 18.
    # If we find one, return it.
    # If not, try 17, etc.
    
    # However, backtracking can be slow if the solution space is sparse.
    # We will use a time-limited backtracking.
    
    # One issue: The order of candidates in the list matters for pruning.
    # If we put "bad" candidates first, we waste time.
    # If we put "good" candidates first, we find solutions faster.
    # What is a "good" candidate? One that doesn't conflict with many others.
    # Primes are generally good.
    
    # Let's order candidates by "complexity" or just use a fixed heuristic order.
    # Heuristic: Primes first, then composites.
    # Or: Larger numbers first?
    
    # Let's try a few different orderings if the first one fails quickly.
    
    candidates_base = list(range(2, 41))
    
    # Ordering 1: Ascending
    # Ordering 2: Descending
    # Ordering 3: Primes first
    
    primes = [p for p in candidates_base if all(p % i != 0 for i in range(2, int(p**0.5)+1))]
    composites = [c for c in candidates_base if c not in primes]
    order_primes_first = primes + composites
    
    # We'll try backtracking with these orders.
    # To be safe, we'll also keep a fallback greedy/local search if backtracking fails?
    # The directive says "different strategy". Backtracking is the main one.
    # But if it times out, we need to return *something*.
    # So we will run backtracking with a time limit.
    # If it doesn't find a large set, we can use a simple greedy to get a valid baseline.
    
    # Let's implement a simple greedy to have a baseline result immediately.
    def greedy_baseline():
        current = []
        vectors = {(0,) * NUM_PRIMES}
        for x in range(2, 41):
            f_x = FACTORS[x]
            valid = True
            for v in vectors:
                nv = tuple(v[i] + f_x[i] for i in range(NUMIMES))
                if nv in vectors:
                    valid = False
                    break
            if valid:
                current.append(x)
                new_vecs = set()
                for v in vectors:
                    nv = tuple(v[i] + f_x[i] for i in range(NUM_PRIMES))
                    new_vecs.add(nv)
                vectors.update(new_vecs)
        return current
    
    # Note: There is a typo in my thought above NUMIMES -> NUM_PRIMES.
    # Let's fix that in the code.
    
    baseline = greedy_baseline()
    best_size = len(baseline)
    best_set = baseline
    
    # Now try to improve with Backtracking.
    # We try to find a set of size best_size + 1, then +2, etc.
    # Or just try for MAX_SIZE and see if we can beat the baseline.
    
    # Actually, backtracking for size 18 might be very hard if it's impossible.
    # We should try sizes in descending order: 18, 17, ...
    # But if 18 is impossible, we waste time.
    # Let's try 18 first with a short time limit (e.g., 10s).
    # If found, return.
    # If not, try 17.
    
    # To speed up backtracking, we can use the fact that we already have a valid set of size K.
    # We can try to extend the greedy set? No, backtracking is from scratch.
    
    # Let's run backtracking for target_size = 18.
    # We will use the "Primes First" ordering as it's often better for independence.
    
    bt = Backtracker()
    
    # We need to be careful with recursion depth. Max depth 18, so it's fine.
    
    # To avoid getting stuck in a specific branch, we can randomize the order slightly?
    # No, deterministic is better for reproducibility, but we can try multiple orderings.
    
    orderings = [
        order_primes_first,
        sorted(candidates_base, reverse=True),
        candidates_base
    ]
    
    found_large = False
    for order in orderings:
        if _time_exceeded():
            break
            
        bt = Backtracker()
        # We want to find a set of size 18.
        # If we find one, we can stop.
        # But what if 18 is not possible? We should try 17.
        
        # Let's try to find a set of size 18.
        # We'll give it a specific time slice.
        # Actually, the _search function checks _time_exceeded globally.
        # We should probably reset the deadline or use a local time limit.
        
        # Let's use a local time limit for each backtracking attempt.
        local_deadline = time.time() + 15.0  # 15 seconds per ordering
        
        # Modify Backtracker to accept a deadline?
        # Or just use the global one and break out if we've spent too much time.
        
        # Let's restructure:
        # Try to find size 18.
        success = bt._search(18, order)
        
        if success:
            if bt.best_size > best_size:
                best_size = bt.best_size
                best_set = bt.best_set
            found_large = True
            break # Found 18, which is max, so we are done
        
        # If not found 18, try 17?
        # It might be that 18 is possible but we didn't find it in this ordering.
        # Or 18 is impossible.
        # Let's try 17 with the same ordering if time permits.
        
        if time.time() < _DEADLINE - 10:
            bt2 = Backtracker()
            success = bt2._search(17, order)
            if success:
                if bt2.best_size > best_size:
                    best_size = bt2.best_size
                    best_set = bt2.best_set
                # Don't break, maybe another ordering finds 18?
                # But 17 is good.
        
    # If we didn't find anything better than the greedy baseline, return the baseline.
    
    # Final check for validity
    if not best_set:
        best_set = [2]
        
    # Ensure types
    return [int(x) for x in best_set]

# Fix the typo in the thought process and ensure the code is correct.
# The greedy_baseline function in the code above had a typo NUMIMES.
# I will write the correct code below.

def construct():
    start_time = time.time()
    
    # Greedy Baseline
    def _greedy():
        current = []
        vectors = set()
        vectors.add((0,) * NUM_PRIMES)
        for x in range(2, MAX_VAL + 1):
            f_x = FACTORS[x]
            valid = True
            for v in vectors:
                nv = tuple(v[i] + f_x[i] for i in range(NUM_PRIMES))
                if nv in vectors:
                    valid = False
                    break
            if valid:
                current.append(x)
                new_vecs = set()
                for v in vectors:
                    nv = tuple(v[i] + f_x[i] for i in range(NUM_PRIMES))
                    new_vecs.add(nv)
                vectors.update(new_vecs)
        return current

    best_set = _greedy()
    best_size = len(best_set)
    
    # Backtracking Improvement
    # We try to find a set larger than best_size.
    # Max size is 18.
    
    candidates_base = list(range(2, MAX_VAL + 1))
    
    # Heuristic ordering: Primes first
    primes = [p for p in candidates_base if all(p % i != 0 for i in range(2, int(p**0.5)+1))]
    composites = [c for c in candidates_base if c not in primes]
    order_primes_first = primes + composites
    
    # Other orderings
    order_desc = sorted(candidates_base, reverse=True)
    order_asc = candidates_base
    
    orderings = [order_primes_first, order_desc, order_asc]
    
    # We will try to find a set of size 18, then 17, etc.
    # But we only care if it's > best_size.
    
    for target_size in range(MAX_SIZE, best_size, -1):
        if time.time() > _DEADLINE - 5:
            break
            
        for order in orderings:
            if time.time() > _DEADLINE - 5:
                break
                
            bt = Backtracker()
            # We want to find ANY set of size target_size.
            # The _search function returns True if found.
            # It updates best_size and best_set in the Backtracker instance.
            
            # We need to pass a time limit to the search to avoid hanging.
            # The global _DEADLINE is used.
            
            found = bt._search(target_size, order)
            
            if found:
                if bt.best_size > best_size:
                    best_size = bt.best_size
                    best_set = bt.best_set
                # If we found the max possible size (18), we can stop immediately.
                if best_size == MAX_SIZE:
                    return [int(x) for x in best_set]
            else:
                # If not found in this ordering, try next ordering?
                # Or next target size?
                # If target_size is 18 and not found, it might be impossible or just hard.
                # Let's try the next ordering for the same target size.
                pass
    
    return [int(x) for x in best_set]

# The Backtracker class needs to be defined before construct.
# I will place it in the final code block.

class Backtracker:
    def __init__(self):
        self.current_set = []
        self.current_vectors = set()
        self.current_vectors.add((0,) * NUM_PRIMES)
        self.best_size = 0
        self.best_set = []
        
    def _add_candidate(self, x):
        f_x = FACTORS[x]
        new_vecs = set()
        for v in self.current_vectors:
            nv = tuple(v[i] + f_x[i] for i in range(NUM_PRIMES))
            if nv in self.current_vectors:
                return False
            new_vecs.add(nv)
        self.current_set.append(x)
        self.current_vectors.update(new_vecs)
        return True

    def _remove_candidate(self, x):
        self.current_set.pop()
        self._recompute_vectors()

    def _recompute_vectors(self):
        if not self.current_set:
            self.current_vectors = {(0,) * NUM_PRIMES}
            return
        vectors = set()
        vectors.add((0,) * NUM_PRIMES)
        for x in self.current_set:
            f_x = FACTORS[x]
            new_vectors = set()
            for v in vectors:
                nv = tuple(v[i] + f_x[i] for i in range(NUM_PRIMES))
                new_vectors.add(nv)
            vectors.update(new_vectors)
        self.current_vectors = vectors

    def _search(self, target_size, candidates):
        if _time_exceeded():
            return False
        if len(self.current_set) == target_size:
            if target_size > self.best_size:
                self.best_size = target_size
                self.best_set = self.current_set[:]
            return True
        
        for i in range(len(candidates)):
            if _time_exceeded():
                return False
            x = candidates[i]
            if self._add_candidate(x):
                remaining = candidates[i+1:]
                if self._search(target_size, remaining):
                    return True
                self._remove_candidate(x)
        return False

# Re-define construct to use the class properly.
# The previous construct definition was inside the thought process.
# I will output the clean version.

def construct():
    start_time = time.time()
    
    def _greedy():
        current = []
        vectors = set()
        vectors.add((0,) * NUM_PRIMES)
        for x in range(2, MAX_VAL + 1):
            f_x = FACTORS[x]
            valid = True
            for v in vectors:
                nv = tuple(v[i] + f_x[i] for i in range(NUM_PRIMES))
                if nv in vectors:
                    valid = False
                    break
            if valid:
                current.append(x)
                new_vecs = set()
                for v in vectors:
                    nv = tuple(v[i] + f_x[i] for i in range(NUM_PRIMES))
                    new_vecs.add(nv)
                vectors.update(new_vecs)
        return current

    best_set = _greedy()
    best_size = len(best_set)
    
    candidates_base = list(range(2, MAX_VAL + 1))
    
    primes = [p for p in candidates_base if all(p % i != 0 for i in range(2, int(p**0.5)+1))]
    composites = [c for c in candidates_base if c not in primes]
    order_primes_first = primes + composites
    
    order_desc = sorted(candidates_base, reverse=True)
    order_asc = candidates_base
    
    orderings = [order_primes_first, order_desc, order_asc]
    
    for target_size in range(MAX_SIZE, best_size, -1):
        if time.time() > _DEADLINE - 5:
            break
        for order in orderings:
            if time.time() > _DEADLINE - 5:
                break
            bt = Backtracker()
            found = bt._search(target_size, order)
            if found:
                if bt.best_size > best_size:
                    best_size = bt.best_size
                    best_set = bt.best_set
                if best_size == MAX_SIZE:
                    return [int(x) for x in best_set]
    
    return [int(x) for x in best_set]

# Ensure the global variables are defined.
MAX_VAL = 40
MAX_SIZE = 18
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
NUM_PRIMES = len(PRIMES)

def _factorize(n):
    exponents = [0] * NUM_PRIMES
    for i, p in enumerate(PRIMES):
        while n % p == 0:
            exponents[i] += 1
            n //= p
    return tuple(exponents)

FACTORS = {i: _factorize(i) for i in range(2, MAX_VAL + 1)}

_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0

def _time_exceeded():
    return time.time() > _DEADLINE

# The Backtracker class and construct function are defined above.
# I need to make sure the order of definitions is correct in the final output.

