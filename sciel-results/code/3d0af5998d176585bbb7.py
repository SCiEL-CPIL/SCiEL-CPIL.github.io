"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

N = 5000

# EVOLVE-BLOCK-START

def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

# Precompute primes up to N+1 (max size could be N+1)
PRIMES = _primes(N + 1)
PRIME_SET = set(PRIMES)

class AdmissibleSet:
    def __init__(self, elements=None):
        self.elements = set(elements) if elements else set()
        self.size = len(self.elements)
        # For each prime p <= self.size, we track the count of elements in each residue class mod p.
        # We only need to ensure that for each p <= size, there is at least one residue class with count 0.
        # Actually, the condition is: for every prime p <= len(tuple), some residue class mod p is missed.
        # "Missed" means count is 0.
        
        # To make this fast, we can maintain for each relevant prime p:
        # covered_residues[p] = set of residues r such that count[r] > 0
        # But updating this on add/remove is tricky if we just have a set.
        
        # Alternative: 
        # For a set S of size k, it is admissible iff for all primes p <= k, |{x % p : x in S}| < p.
        # This is equivalent to saying the set of residues covered mod p has size < p.
        
        # Let's maintain for each prime p in PRIMES (where p <= N+1):
        # residue_counts[p] = [0] * p
        # And a counter: empty_count[p] = number of residues with count 0.
        # Initially, for a given p, empty_count[p] = p.
        # When we add x, for each p <= new_size, we update residue_counts[p][x%p].
        # If residue_counts[p][r] becomes 1 (from 0), empty_count[p] decreases by 1.
        # If empty_count[p] becomes 0, the set is NOT admissible (for that p).
        
        # However, we only care about p <= self.size.
        # As size changes, the set of relevant primes changes.
        
        # Optimization: 
        # We can just check admissibility on the fly for candidate additions/removals.
        # But we need to do this fast.
        
        # Let's try a different representation for speed:
        # We maintain the set of elements.
        # To check if adding x is valid for current size k (new size k+1):
        # We need to check all primes p <= k+1.
        # For each p, we need to know if x % p is already "covered" in a way that would leave no empty residue?
        # No, we need to know if x % p is the LAST empty residue for p.
        
        # Let's maintain:
        # For each prime p, a set `covered[p]` of residues mod p that are present in the set.
        # Then admissibility for size k is: for all p <= k, len(covered[p]) < p.
        # Adding x: for all p <= k+1, if x%p not in covered[p], add it.
        # Check: for all p <= k+1, len(covered[p]) < p.
        # This is O(pi(k)) per operation. pi(5000) is approx 669.
        # 5000 elements * 669 operations * log(N) overhead... might be too slow for 300s if we do many restarts.
        
        # Let's optimize the check.
        # We only need to track primes p <= current_size.
        # We can maintain a dictionary `residue_sets` mapping p -> set of residues.
        
        self.residue_sets = {}
        
        if elements:
            for x in elements:
                self._add_internal(x)
    
    def _get_p_limit(self, size):
        # Return primes up to size
        # Since size changes, we can't precompute a fixed list easily without binary search.
        # But we can just iterate through PRIMES and break when p > size.
        pass

    def _add_internal(self, x):
        self.elements.add(x)
        self.size += 1
        # Update residue sets for all primes p <= self.size
        for p in PRIMES:
            if p > self.size:
                break
            if p not in self.residue_sets:
                self.residue_sets[p] = set()
            self.residue_sets[p].add(x % p)
            
    def _remove_internal(self, x):
        # We need to rebuild residue sets or update them carefully.
        # Removing is hard if we just have sets. 
        # If we remove x, we need to know if x%p was the ONLY element with that residue.
        # So we need counts.
        
        # Let's switch to counts.
        # This class is getting complicated. Let's restart the design inside construct.
        pass

def construct():
    deadline = time.time() + 290
    best_set = set()
    best_size = 0
    
    # Precompute primes
    primes = PRIMES
    prime_set = PRIME_SET
    
    # Helper to check if a set is admissible
    # This is O(k * pi(k)) which is too slow for repeated checks.
    
    # Better approach:
    # Maintain state incrementally.
    
    class State:
        def __init__(self):
            self.elements = set()
            self.size = 0
            # For each prime p, we maintain a list of counts for residues 0..p-1
            # And a count of how many residues are empty (count 0).
            # We only maintain this for primes p <= self.size.
            # When size increases, we add new primes.
            # When size decreases, we don't necessarily remove primes, but we only check p <= size.
            
            self.res_counts = {} # p -> list of counts
            self.empty_counts = {} # p -> number of residues with count 0
            
        def _ensure_prime(self, p):
            if p not in self.res_counts:
                # Initialize for this prime
                # We need to compute counts from scratch for existing elements
                counts = [0] * p
                for x in self.elements:
                    counts[x % p] += 1
                self.res_counts[p] = counts
                self.empty_counts[p] = sum(1 for c in counts if c == 0)
                
        def add(self, x):
            if x in self.elements:
                return False
            
            new_size = self.size + 1
            
            # Check validity for all primes p <= new_size
            # We need to ensure that after adding x, for all p <= new_size, empty_count[p] > 0.
            
            # First, ensure all primes up to new_size are initialized in our tracking structures
            # Note: adding x might make a prime p "critical" if it was not previously tracked (p > old_size)
            
            # We can just iterate through primes <= new_size
            valid = True
            for p in primes:
                if p > new_size:
                    break
                
                if p not in self.res_counts:
                    # Initialize
                    self._ensure_prime(p)
                
                r = x % p
                count = self.res_counts[p][r]
                
                if count == 0:
                    # This residue was empty. Adding x makes it non-empty.
                    # So empty_count decreases by 1.
                    if self.empty_counts[p] == 1:
                        # This was the last empty residue. Adding x would make empty_count 0.
                        # Invalid.
                        valid = False
                        break
                
            if not valid:
                return False
            
            # Perform the addition
            self.elements.add(x)
            self.size = new_size
            
            for p in primes:
                if p > new_size:
                    break
                r = x % p
                self.res_counts[p][r] += 1
                if self.res_counts[p][r] == 1:
                    self.empty_counts[p] -= 1
                    
            return True
        
        def remove(self, x):
            if x not in self.elements:
                return False
            
            old_size = self.size
            new_size = old_size - 1
            
            # Check validity for all primes p <= new_size
            # Note: for p > new_size, we don't care about admissibility anymore.
            # But we do need to update counts.
            
            # Actually, we just need to ensure that for p <= new_size, empty_count[p] > 0.
            # Removing x can only INCREASE empty_count (if it was the last one).
            # So removal is always valid with respect to the "empty" condition?
            # Wait. The condition is "some residue class is missed".
            # If we remove x, we might make a residue class empty that was not.
            # This helps admissibility.
            # Can removal make it INVALID?
            # No. If it was admissible before, removing an element cannot make a previously non-empty residue class become "required to be empty".
            # The condition is: for all p <= k, there exists r such that count[r] == 0.
            # Removing an element can only turn a count from 1 to 0 (increasing empty count) or 2 to 1 (no change in empty count).
            # It cannot turn a 0 into a 1.
            # So empty_count[p] can only stay same or increase.
            # If empty_count[p] > 0 before, it will be > 0 after.
            # So removal is ALWAYS valid for the admissibility constraint of the NEW size?
            # Yes, because the set of primes we check shrinks (p <= new_size) and the empty counts for those primes can only increase.
            
            # So we can just remove.
            
            self.elements.discard(x)
            self.size = new_size
            
            # Update counts
            # We only need to update primes p <= old_size (since they were tracked)
            # But we should also update p <= new_size for consistency if we do further adds.
            # Actually, we should update all tracked primes.
            for p in self.res_counts:
                # We only care about updating if p <= old_size? 
                # The counts for p > new_size are not used for validity checks anymore, 
                # but if we add again, size increases, and we might need them.
                # So we should update all tracked primes.
                r = x % p
                self.res_counts[p][r] -= 1
                if self.res_counts[p][r] == 0:
                    self.empty_counts[p] += 1
            
            return True

    # Strategy: Multi-start Randomized Local Search
    # 1. Generate a random initial valid set.
    #    How? Start empty, add random elements if valid.
    # 2. Local Search: 
    #    - Try to add a random element if valid.
    #    - If stuck, try to remove a random element, then add two different elements?
    #    - Or just try to add random elements. If none work, remove one and try again.
    
    def random_init(state, deadline):
        # Start with empty set
        state.elements = set()
        state.size = 0
        state.res_counts = {}
        state.empty_counts = {}
        
        # Add random elements
        candidates = list(range(N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            if state.add(x):
                pass
            # If add fails, just skip. We want a large set, so we try many.
        return state.size

    def local_search(state, deadline):
        # Try to improve the set
        # Loop until deadline or no improvement for a while?
        # Since we have a fixed deadline, just run until deadline.
        
        candidates = list(range(N + 1))
        
        while time.time() < deadline:
            random.shuffle(candidates)
            improved = False
            
            for x in candidates:
                if time.time() > deadline:
                    break
                
                if x not in state.elements:
                    if state.add(x):
                        improved = True
                        # If we improved, we can restart the shuffle or continue?
                        # Continuing is fine, but the set of valid additions changes.
                        # Let's just continue scanning.
                    else:
                        # Try to remove a random element and then add x?
                        # This is a "swap" move.
                        # But removal is always valid.
                        # So: if we can't add x, maybe we can remove some y and add x?
                        # This is a 1-for-1 swap.
                        # But we want to INCREASE size.
                        # So a swap doesn't increase size.
                        # We need to find a way to add x.
                        # The only way to add x if it's currently invalid is to remove some element(s)
                        # that are blocking it.
                        # But removing elements decreases size.
                        # So we would need to add x AND some other element z to net gain?
                        # Or just accept that we are stuck and try a different x.
                        
                        pass
            
            if not improved:
                # We couldn't add any new element in this pass.
                # We are in a local maximum.
                # To escape, we can remove a random element and try to add 2 new ones?
                # Or just remove one and try to add 1 (net 0, but might unblock a larger addition later).
                # Let's try: Remove a random element, then try to add random elements until size is restored or improved.
                
                if state.size > 0:
                    to_remove = random.choice(list(state.elements))
                    state.remove(to_remove)
                    
                    # Now try to add elements to regain size and potentially more
                    # We have a "hole" of size 1.
                    # Try to add 2 elements to get net +1.
                    added_count = 0
                    for y in candidates:
                        if time.time() > deadline:
                            break
                        if y not in state.elements:
                            if state.add(y):
                                added_count += 1
                                if added_count >= 2:
                                    # We regained size and gained 1
                                    break
                    # If we only added 1, we are back to same size but different composition.
                    # If we added 0, we are at size-1. We should try to add 1 more to get back.
                    if added_count < 2:
                         for y in candidates:
                             if time.time() > deadline:
                                 break
                             if y not in state.elements:
                                 if state.add(y):
                                     break

    state = State()
    
    # Run multiple restarts? 
    # The local search itself runs until deadline.
    # But maybe a better initial state helps?
    # Let's just do one long local search run with the "escape" mechanism.
    
    # Actually, the "escape" mechanism (remove 1, add 2) is a standard way to escape local maxima in this kind of problem.
    
    # Let's refine the loop:
    # 1. Init
    # 2. Loop:
    #    a. Try to add any valid x.
    #    b. If none found, perform a "perturbation": remove random y, then try to add randoms until size >= old_size.
    
    random.seed(42)
    
    random_init(state, deadline)
    current_best = state.size
    
    while time.time() < deadline:
        # Try to add
        added_any = False
        candidates = list(range(N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            if x not in state.elements:
                if state.add(x):
                    added_any = True
                    current_best = state.size
                    # Don't break, keep trying to add more
                    # But we need to reshuffle? No, the list is fixed.
                    # Just continue.
        
        if not added_any:
            # Local max. Perturb.
            if state.size > 0:
                to_remove = random.choice(list(state.elements))
                state.remove(to_remove)
                
                # Try to add back and more
                # We want to reach size > current size eventually.
                # Let's try to add 2 elements.
                added_back = 0
                for y in candidates:
                    if time.time() > deadline:
                        break
                    if y not in state.elements:
                        if state.add(y):
                            added_back += 1
                            if added_back >= 2:
                                break
                
                # If we failed to add 2, we might be at size-1 or size.
                # If we are at size-1, we should try to add 1 more to get back to size.
                if added_back < 2:
                    for y in candidates:
                        if time.time() > deadline:
                            break
                        if y not in state.elements:
                            if state.add(y):
                                break
    
    return list(state.elements)

# EVOLVE-BLOCK-END
