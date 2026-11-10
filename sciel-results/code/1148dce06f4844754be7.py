"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s limit for safety margin
    
    n = 256
    all_elements = list(range(1, n + 1))
    recips = {x: Fraction(1, x) for x in all_elements}
    
    # Helper to check if adding x creates a 3-term AP with existing set
    # A 3-term AP is a, a+d, a+2d. 
    # If we have a set S, adding x is invalid if there exist a, b in S such that:
    # 1. x = 2b - a (x is the end)
    # 2. b = 2x - a (x is the middle) => a = 2x - b
    # 3. a = 2x - b (x is the start) => b = 2x - a
    # Essentially, for any pair (a, b) in S, we cannot add x if x is in {2b-a, (a+b)/2 (if int), 2a-b}
    # But simpler: just check all pairs in S for APs involving x.
    
    def is_valid_addition(x, current_set):
        # Check if x forms a 3-AP with any two elements in current_set
        # Let the AP be (y, z, w) with y < z < w.
        # If x is in the AP, it's either y, z, or w.
        # If x is w: need y, z in S such that 2z - y = x => y = 2z - x.
        # If x is z: need y, w in S such that y + w = 2x.
        # If x is y: need z, w in S such that 2z - w = x => w = 2z - x.
        
        # Since S is a set, we can iterate through S.
        # Optimization: Pre-compute conflicts? 
        # Given n=256 and set size ~50-100, iterating pairs is O(k^2) ~ 10^4 ops. 
        # This is fast enough for many iterations.
        
        for y in current_set:
            # Case 1: x is the largest term (w). y is middle (z). Need start (a) = 2y - x.
            a = 2 * y - x
            if a in current_set and a != y:
                return False
            
            # Case 2: x is the middle term (z). y is one end. Need other end (b) = 2x - y.
            b = 2 * x - y
            if b in current_set and b != x and b != y:
                return False
            
            # Case 3: x is the smallest term (y). y is... wait, y is from S.
            # If x is smallest, then y is middle (z) or largest (w).
            # If y is middle (z): need largest (w) = 2y - x. (Same as Case 1 but roles of x,y swapped in logic? No.)
            # Let's re-verify.
            # AP: (x, y, w) => 2y = x + w => w = 2y - x. (x is start, y is mid)
            # AP: (x, w, y) => 2w = x + y => w = (x+y)/2. (x is start, y is end) -> w must be int.
            # AP: (y, x, w) => 2x = y + w => w = 2x - y. (y is start, x is mid) -> Covered in Case 2.
            # AP: (w, x, y) => 2x = w + y => w = 2x - y. (y is end, x is mid) -> Covered in Case 2.
            # AP: (y, w, x) => 2w = y + x => w = (x+y)/2. (y is start, x is end)
            # AP: (w, y, x) => 2y = w + x => w = 2y - x. (y is mid, x is end) -> Covered in Case 1.
            
            # So we need to check:
            # 1. w = 2y - x is in S (y is mid, x is end, OR y is mid, x is start? No, if x is start, y is mid, w is end. 2y = x+w => w=2y-x. Yes.)
            # 2. w = (x+y)/2 is in S (if x+y is even) (x is start, y is end, w is mid. OR y is start, x is end, w is mid.)
            
            if (x + y) % 2 == 0:
                w = (x + y) // 2
                if w in current_set and w != x and w != y:
                    return False
        
        return True

    def get_score(s):
        return sum(recips[x] for x in s)

    # Initialize with a random valid set
    # We can start with a greedy set on a random permutation to get a diverse starting point
    def random_greedy_init():
        perm = all_elements[:]
        random.shuffle(perm)
        s = []
        for x in perm:
            if is_valid_addition(x, s):
                s.append(x)
        return s

    best_set = random_greedy_init()
    best_score = get_score(best_set)
    current_set = best_set[:]
    current_score = best_score
    
    # Local Search: Simulated Annealing / Hill Climbing
    # Moves:
    # 1. Insert a random element not in set.
    # 2. Remove a random element from set.
    # 3. Swap a random element in set with a random element not in set.
    
    # To make it efficient, we can maintain a list of "candidates" or just pick randomly.
    # Given n=256, picking random is fine.
    
    in_set = set(current_set)
    not_in_set = [x for x in all_elements if x not in in_set]
    
    temp = 1.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Pick a move
        move_type = random.choice(['insert', 'remove', 'swap'])
        
        new_set = current_set[:]
        new_in_set = in_set.copy()
        
        if move_type == 'insert':
            if not not_in_set:
                continue
            x = random.choice(not_in_set)
            if is_valid_addition(x, new_set):
                new_set.append(x)
                new_in_set.add(x)
            else:
                continue # Invalid move, skip
        elif move_type == 'remove':
            if not current_set:
                continue
            x = random.choice(current_set)
            new_set.remove(x)
            new_in_set.discard(x)
        else: # swap
            if not current_set or not not_in_set:
                continue
            x_out = random.choice(current_set)
            x_in = random.choice(not_in_set)
            # Try swapping: remove x_out, add x_in
            temp_set = [x for x in new_set if x != x_out]
            if is_valid_addition(x_in, temp_set):
                new_set = temp_set + [x_in]
                new_in_set.discard(x_out)
                new_in_set.add(x_in)
            else:
                continue

        new_score = get_score(new_set)
        delta = new_score - current_score
        
        if delta > 0:
            # Accept
            current_set = new_set
            current_score = new_score
            in_set = new_in_set
            not_in_set = [x for x in all_elements if x not in in_set]
            
            if current_score > best_score:
                best_set = current_set[:]
                best_score = current_score
        else:
            # Accept with probability exp(delta / temp)
            # delta is negative
            if temp > 0.001:
                prob = 2.718281828 ** (delta / temp)
                if random.random() < prob:
                    current_set = new_set
                    current_score = new_score
                    in_set = new_in_set
                    not_in_set = [x for x in all_elements if x not in in_set]
        
        temp *= cooling
        
    # Final check: ensure best_set is valid (should be, but just in case)
    # The logic above ensures validity at every step.
    
    return best_set
# EVOLVE-BLOCK-END
