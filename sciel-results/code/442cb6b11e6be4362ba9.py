import time
import random
import sys
from itertools import combinations_with_replacement

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill

    n_max = 4096
    max_count = 80

    def can_add(selected, candidate):
        """Check if adding candidate to selected keeps all triple sums unique."""
        # We need to check:
        # 1. All triples within selected are already unique (assumed)
        # 2. New triples involving candidate don't collide with existing or each other
        
        # New triples are of the form:
        # (candidate, candidate, candidate) -> sum = 3*candidate
        # (candidate, candidate, x) for x in selected -> sum = 2*candidate + x
        # (candidate, x, y) for x <= y in selected -> sum = candidate + x + y
        
        # We need to check these new sums don't collide with:
        # a) Each other
        # b) Existing sums (triples entirely within selected)
        
        # Optimization: Pre-compute existing sums as a set
        # But that's expensive for large selected. Instead, let's be smarter.
        
        # For a set S of size k, there are C(k+2,3) = (k+2)(k+1)k/6 triples.
        # For k=80, that's about 91390 triples. Checking collisions for each new candidate
        # by building the full set is O(k^3) which is too slow.
        
        # Better approach: For the new element c, the new sums are:
        # S_new = {3c} ∪ {2c+x : x ∈ S} ∪ {c+x+y : x≤y, x,y ∈ S}
        # We need S_new ∩ S_old = ∅ and S_new has all distinct elements.
        
        # S_old is the set of all triple sums from S.
        # Computing S_old is O(k^3) which for k=80 is ~85k operations, acceptable per check?
        # 85k * 4096 candidates = 350M operations, might be too slow in Python.
        
        # Alternative: Incremental approach. Keep a set of all current sums.
        # When adding c, generate only the new sums (O(k^2)) and check against existing set.
        # Then update the set if valid.
        
        # Let's use this approach:
        # Maintain 'all_sums' as a set of all triple sums for current selected.
        # To test candidate c:
        #   new_sums = set()
        #   new_sums.add(3*c)
        #   for x in selected: new_sums.add(2*c+x)
        #   for i, x in enumerate(selected):
        #       for y in selected[i:]: new_sums.add(c+x+y)
        #   if new_sums & all_sums: return False
        #   if len(new_sums) < len(new_sums): return False  # internal collision
        #   Actually check len(new_sums) == expected count
        #   If OK, update all_sums |= new_sums, add c to selected
        
        return None  # We'll implement in the main loop with shared state
    
    # Better design: use a helper class or closure with shared state
    def try_construct(seed, max_iter=1000000):
        selected = []
        all_sums = set()
        
        for c in range(1, n_max + 1):
            if len(selected) >= max_count:
                break
            if time.time() > deadline:
                break
                
            # Generate new sums involving c
            new_sums = set()
            new_sums.add(3 * c)
            for x in selected:
                new_sums.add(2 * c + x)
            for i in range(len(selected)):
                for j in range(i, len(selected)):
                    new_sums.add(c + selected[i] + selected[j])
            
            # Check for collisions with existing sums
            if new_sums & all_sums:
                continue
            
            # Check internal collisions in new_sums
            expected_count = 1 + len(selected) + len(selected)*(len(selected)+1)//2
            if len(new_sums) < expected_count:
                continue
            
            # Valid! Add c
            selected.append(c)
            all_sums |= new_sums
        
        return selected
    
    # Strategy: Try greedy from 1, then try different starting points or random orders
    best = []
    
    # First, try the standard greedy
    result = try_construct(0)
    if len(result) > len(best):
        best = result[:]
    
    # Try some randomized restarts with different seeds
    for trial in range(50):
        if time.time() > deadline - 5:
            break
        
        # Random subset approach: pick random candidates and try to build
        # Or: start from a random offset
        random.seed(42 + trial)
        # Try skipping some small numbers
        skip_set = set(random.sample(range(1, 50), min(10, trial + 1)))
        
        selected = []
        all_sums = set()
        for c in range(1, n_max + 1):
            if len(selected) >= max_count:
                break
            if time.time() > deadline - 2:
                break
            if c in skip_set:
                continue
                
            new_sums = set()
            new_sums.add(3 * c)
            for x in selected:
                new_sums.add(2 * c + x)
            for i in range(len(selected)):
                for j in range(i, len(selected)):
                    new_sums.add(c + selected[i] + selected[j])
            
            if new_sums & all_sums:
                continue
            
            expected_count = 1 + len(selected) + len(selected)*(len(selected)+1)//2
            if len(new_sums) < expected_count:
                continue
            
            selected.append(c)
            all_sums |= new_sums
        
        if len(selected) > len(best):
            best = selected[:]
    
    return best

# Keep original entry point
def construct():
    return _construct_impl()

def _construct_impl():
    start_time = time.time()
    deadline = start_time + 85
    
    n_max = 4096
    max_count = 80
    
    def greedy_with_skip(skip_set):
        selected = []
        all_sums = set()
        for c in range(1, n_max + 1):
            if len(selected) >= max_count:
                break
            if time.time() > deadline:
                break
            if c in skip_set:
                continue
            
            new_sums = set()
            new_sums.add(3 * c)
            for x in selected:
                new_sums.add(2 * c + x)
            for i in range(len(selected)):
                for j in range(i, len(selected)):
                    new_sums.add(c + selected[i] + selected[j])
            
            if new_sums & all_sums:
                continue
            
            expected_count = 1 + len(selected) + len(selected)*(len(selected)+1)//2
            if len(new_sums) < expected_count:
                continue
            
            selected.append(c)
            all_sums |= new_sums
        return selected
    
    # Standard greedy
    best = greedy_with_skip(set())
    
    # Randomized restarts
    random.seed(123)
    for trial in range(100):
        if time.time() > deadline - 3:
            break
        
        # Randomly skip some small numbers to break symmetry
        skip_size = random.randint(0, 15)
        skip_set = set(random.sample(range(1, 60), min(skip_size, 59)))
        
        result = greedy_with_skip(skip_set)
        if len(result) > len(best):
            best = result
    
    return best
