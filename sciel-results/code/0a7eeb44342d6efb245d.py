"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 55.0  # Safety margin before 90s limit
    best_sets = ([], [])
    best_size = 0
    
    def _validate_sidon(S):
        """Check if S is a Sidon set (unique pair sums including a+a)."""
        if not S:
            return True
        sums = set()
        n = len(S)
        for i in range(n):
            for j in range(i, n):
                s = S[i] + S[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    def _diffs_of(S):
        """Get nonzero differences of S."""
        d = set()
        n = len(S)
        for i in range(n):
            for j in range(n):
                if i != j:
                    d.add(S[j] - S[i])
        return d
    
    def _greedy_one_ordering(order, deadline):
        """Run greedy construction with a given ordering of candidates."""
        sets = [[], []]
        set_sums = [set(), set()]
        set_diffs = [set(), set()]
        
        for x in order:
            if time.time() > deadline:
                break
            
            for side in (0, 1):
                # Check if x can be added to sets[side]
                S = sets[side]
                other_diffs = set_diffs[1 - side]
                
                # Check Sidon condition: new pair sums must be unique and not conflict
                new_sums_valid = True
                for a in S:
                    s = a + x
                    if s in set_sums[side]:
                        new_sums_valid = False
                        break
                    # Also check x + x
                if new_sums_valid:
                    s_xx = x + x
                    if s_xx in set_sums[side]:
                        new_sums_valid = False
                
                if not new_sums_valid:
                    continue
                
                # Check difference disjointness
                new_diffs = set()
                for a in S:
                    new_diffs.add(x - a)
                    new_diffs.add(a - x)
                
                if new_diffs & other_diffs:
                    continue
                
                # Accept x into sets[side]
                sets[side].append(x)
                for a in S:
                    set_sums[side].add(a + x)
                    set_diffs[side].add(x - a)
                    set_diffs[side].add(a - x)
                set_sums[side].add(x + x)
        
        return sets
    
    # Try multiple random orderings
    random.seed(42)
    base_order = list(range(1, 257))
    
    # First try natural order
    sets = _greedy_one_ordering(base_order, deadline)
    size = len(sets[0]) + len(sets[1])
    if size > best_size:
        best_size = size
        best_sets = sets
    
    # Try several random permutations
    num_trials = 0
    while time.time() < deadline - 2.0:
        random_order = base_order[:]
        random.shuffle(random_order)
        
        sets = _greedy_one_ordering(random_order, deadline)
        size = len(sets[0]) + len(sets[1])
        if size > best_size:
            best_size = size
            best_sets = sets
        
        num_trials += 1
        # Also try alternating random orders
        if num_trials % 2 == 0:
            random_order2 = base_order[:]
            random.shuffle(random_order2)
            sets = _greedy_one_ordering(random_order2, deadline)
            size = len(sets[0]) + len(sets[1])
            if size > best_size:
                best_size = size
                best_sets = sets
            num_trials += 1
    
    # Ensure nonempty
    if not best_sets[0] and not best_sets[1]:
        best_sets = ([1], [2])
    elif not best_sets[0]:
        best_sets[0] = [1]
    elif not best_sets[1]:
        best_sets[1] = [1]
    
    return best_sets
# EVOLVE-BLOCK-END
