import random
import math
from itertools import combinations

def construct():
    def is_valid_add(S, x):
        """
        Check if adding x to valid set S keeps it valid.
        S is a list of distinct integers. x is not in S.
        Returns True if S U {x} is valid.
        """
        # 1. Check if x divides sum of any 3 distinct elements in S
        for b, c, d in combinations(S, 3):
            if (b + c + d) % x == 0:
                return False
        
        # 2. Check if any a in S divides sum of any triple containing x
        # Triples containing x are {x, b, c} where b, c are distinct elements from S \ {a}
        # So for each a in S, we check all pairs (b, c) from S such that b != a, c != a, b != c
        # If (x + b + c) % a == 0, invalid.
        for a in S:
            for b in S:
                if b == a:
                    continue
                for c in S:
                    if c == a or c <= b: # Ensure b < c and c != a
                        continue
                    if (x + b + c) % a == 0:
                        return False
        return True

    # Greedy construction
    # Start with empty set
    selected = []
    # Try adding numbers from large to small
    # Large numbers are less likely to divide sums of smaller numbers
    for x in range(192, 0, -1):
        if is_valid_add(selected, x):
            selected.append(x)
    
    best = list(selected)
    best_size = len(best)
    
    # Local Search / Simulated Annealing
    current = list(selected)
    current_size = len(current)
    
    T = 5.0
    cooling = 0.999
    iterations = 20000 # Can be higher now due to faster check
    
    random.seed(42)
    
    for it in range(iterations):
        move_type = random.random()
        
        if move_type < 0.6:
            # Try to add a number
            candidates = [x for x in range(1, 193) if x not in current]
            if not candidates:
                continue
            x = random.choice(candidates)
            if is_valid_add(current, x):
                new_state = current + [x]
                new_size = len(new_state)
                # Since we only add valid elements, size increases by 1
                # Accept if it improves best, or with some probability to allow exploration?
                # Actually, adding a valid element always increases size.
                # So we should always accept if we want to maximize size.
                # But maybe adding x prevents adding y later?
                # In SA, we might accept worse states. But here "worse" means smaller size.
                # Adding a valid element makes size larger. So it's always an improvement in size.
                # However, it might block future additions.
                # So we should accept it, but maybe keep a history or use SA to undo if stuck.
                # For now, always accept valid additions.
                current = new_state
                current_size = new_size
                if current_size > best_size:
                    best = list(current)
                    best_size = current_size
            T *= cooling
            
        elif move_type < 0.8:
            # Remove a random element to allow rearrangement
            if not current:
                continue
            idx = random.randint(0, len(current) - 1)
            removed = current[idx]
            new_state = current[:idx] + current[idx+1:]
            new_size = len(new_state)
            # Removing an element always results in a valid set (subset of valid set)
            # But size decreases.
            # Accept with probability exp(-1/T)
            if random.random() < math.exp(-1 / T):
                current = new_state
                current_size = new_size
            T *= cooling
            
        else:
            # Swap: Remove one, try to add another
            if not current:
                continue
            idx = random.randint(0, len(current) - 1)
            removed = current[idx]
            temp = current[:idx] + current[idx+1:]
            candidates = [x for x in range(1, 193) if x not in temp]
            if candidates:
                x = random.choice(candidates)
                if is_valid_add(temp, x):
                    new_state = temp + [x]
                    new_size = len(new_state)
                    # Size stays same
                    # Accept with probability 1 (neutral) or based on some heuristic?
                    # For now, accept to explore state space
                    current = new_state
                    current_size = new_size
                    if current_size > best_size:
                        best = list(current)
                        best_size = current_size
                T *= cooling
            else:
                T *= cooling

    return best
