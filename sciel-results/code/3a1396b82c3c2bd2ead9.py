import time
import random
from itertools import combinations

# Global deadline
START_TIME = time.time()
TIME_LIMIT = 50.0 # Safety margin for 60s limit

class ValidSet:
    def __init__(self):
        self.S = set()
        self.sums3 = set()
        self.mod_counts = {}
        
    def add(self, x):
        """Add x to the set, updating auxiliary structures."""
        if x in self.S:
            return
        
        # Update mod_counts for existing elements a in S
        for a in self.S:
            if a not in self.mod_counts:
                self.mod_counts[a] = {}
            r = x % a
            self.mod_counts[a][r] = self.mod_counts[a].get(r, 0) + 1
            
        # Initialize mod_counts for x
        self.mod_counts[x] = {0: 1} # x % x == 0
        for v in self.S:
            r = v % x
            self.mod_counts[x][r] = self.mod_counts[x].get(r, 0) + 1
            
        # Update sums3
        # New sums are x + b + c where b, c in S_old, b != c
        S_list = list(self.S)
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                b = S_list[i]
                c = S_list[j]
                self.sums3.add(x + b + c)
                
        self.S.add(x)

    def remove(self, x):
        """Remove x from the set, updating auxiliary structures."""
        if x not in self.S:
            return
            
        # Update mod_counts for existing elements a in S \ {x}
        for a in self.S:
            if a == x:
                continue
            r = x % a
            if r in self.mod_counts[a]:
                self.mod_counts[a][r] -= 1
                if self.mod_counts[a][r] == 0:
                    del self.mod_counts[a][r]
                    
        # Remove mod_counts for x
        if x in self.mod_counts:
            del self.mod_counts[x]
            
        # Update sums3
        # Remove sums involving x: x + b + c where b, c in S \ {x}, b != c
        S_list = list(self.S)
        S_list.remove(x)
        sums_to_remove = set()
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                b = S_list[i]
                c = S_list[j]
                sums_to_remove.add(x + b + c)
                
        for s in sums_to_remove:
            if s in self.sums3:
                self.sums3.remove(s)
                
        self.S.remove(x)

    def can_add(self, x):
        """Check if x can be added to the current set."""
        if x in self.S:
            return False
            
        # Check 1: x does not divide any sum of 3 distinct elements in S
        for s in self.sums3:
            if s % x == 0:
                return False
                
        # Check 2: No a in S divides sum of x and any 2 distinct elements in S \ {a}
        S_list = list(self.S)
        n = len(S_list)
        
        for a in S_list:
            # We need to check if there exist b, c in S \ {a}, b != c
            # such that (x + b + c) % a == 0
            # This is equivalent to c % a == (-x - b) % a
            # Let r_target = (-x - b) % a
            # We need to know if there is c in S \ {a, b} with c % a == r_target
            
            # Get counts for modulus a
            counts_a = self.mod_counts.get(a, {})
            
            for b in S_list:
                if b == a:
                    continue
                
                r_target = (-x - b) % a
                
                # How many elements in S have remainder r_target mod a?
                cnt = counts_a.get(r_target, 0)
                
                if cnt == 0:
                    continue
                
                # We need c in S \ {a, b}
                # The count `cnt` includes all elements in S with that remainder.
                # Note: a % a == 0. If r_target == 0, 'a' is counted in cnt.
                # But we are iterating b from S, and checking existence of c.
                # The set of candidates for c is S \ {a, b}.
                
                # Let's refine the count.
                # Total elements in S with rem r_target: cnt
                # If a % a == r_target (i.e., r_target == 0), then 'a' is one of them.
                # If b % a == r_target, then 'b' is one of them.
                
                valid_candidates = cnt
                
                if r_target == 0:
                    valid_candidates -= 1 # 'a' is not allowed as c
                
                if b % a == r_target:
                    valid_candidates -= 1 # 'b' is not allowed as c
                
                if valid_candidates > 0:
                    return False
                    
        return True

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_set = set()
    best_size = 0
    
    numbers = list(range(1, 193))
    
    while time.time() - START_TIME < TIME_LIMIT:
        # Random restart
        vs = ValidSet()
        
        # Greedy phase with random order, biased towards larger numbers
        # Larger numbers are less likely to divide sums
        candidates = numbers.copy()
        random.shuffle(candidates)
        # Sort by descending value to try larger ones first? 
        # Or just random. Let's mix.
        # Actually, starting with large numbers is a good heuristic.
        candidates.sort(reverse=True)
        # Randomize slightly to avoid local optima
        for i in range(len(candidates) - 1):
            j = random.randint(i, len(candidates) - 1)
            candidates[i], candidates[j] = candidates[j], candidates[i]
            
        for x in candidates:
            if vs.can_add(x):
                vs.add(x)
                
        # Local search: try to improve
        # 1. Try adding any missing number
        # 2. Try swapping
        improved = True
        while improved and time.time() - START_TIME < TIME_LIMIT:
            improved = False
            
            # Try adding
            missing = [x for x in numbers if x not in vs.S]
            random.shuffle(missing)
            for x in missing:
                if vs.can_add(x):
                    vs.add(x)
                    improved = True
                    break
            
            if not improved:
                # Try swapping
                # Pick a random element to remove, try to add a random missing element
                # Or iterate.
                S_list = list(vs.S)
                if S_list:
                    random.shuffle(S_list)
                    for y in S_list:
                        # Try removing y
                        vs.remove(y)
                        
                        # Try adding any missing
                        for x in missing:
                            if x == y:
                                continue
                            if vs.can_add(x):
                                vs.add(x)
                                improved = True
                                break
                        if improved:
                            break
                        
                        if not improved:
                            # Re-add y if we didn't find a swap
                            vs.add(y)
                        
                        if time.time() - START_TIME > TIME_LIMIT - 5:
                            break
                        
                    if time.time() - START_TIME > TIME_LIMIT - 5:
                        break

        if len(vs.S) > best_size:
            best_size = len(vs.S)
            best_set = set(vs.S)
            
    return list(best_set)
