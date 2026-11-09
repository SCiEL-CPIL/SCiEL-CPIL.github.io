import random
import math

def construct():
    N = 256
    
    # Precompute conflicts for fast checking
    # conflict[x] is a list of (y, z) such that x, y, z form an AP?
    # Actually, we need to check if adding x creates an AP with any pair (a, b) in S.
    # Or if x completes an AP (a, x, b) or (a, b, x) etc.
    
    # Let's use a bitset for the set for speed if possible, but N=256 fits in 4 64-bit ints.
    # Python sets are okay for N=256.
    
    # Precompute all 3-term APs in [1, N]
    # There are roughly N^2 / 2 such triples.
    # For each triple (a, b, c), if we pick two, we cannot pick the third.
    
    # We will use Simulated Annealing.
    
    # Initial state: Greedy construction (same as seed's greedy, but we treat it as start)
    in_set = [False] * (N + 1)
    current_set = set()
    
    # Greedy init
    for x in range(1, N + 1):
        valid = True
        for a in current_set:
            # Check if a, x, ? or ?, a, x or a, ?, x forms AP
            # 1. a, x, b => b = 2x - a. If b in current_set and b != x and b != a
            # 2. a, b, x => b = (a+x)/2. If a+x even, b=(a+x)//2. If b in current_set and b != a and b != x
            # 3. x, a, b => b = 2a - x. If b in current_set and b != x and b != a
            
            b1 = 2 * x - a
            if 1 <= b1 <= N and b1 != x and b1 != a and b1 in current_set:
                valid = False
                break
            
            s = a + x
            if s % 2 == 0:
                b2 = s // 2
                if b2 != a and b2 != x and b2 in current_set:
                    valid = False
                    break
            
            b3 = 2 * a - x
            if 1 <= b3 <= N and b3 != x and b3 != a and b3 in current_set:
                valid = False
                break
        
        if valid:
            in_set[x] = True
            current_set.add(x)
            
    def get_score(s):
        return sum(1.0/i for i in s)
        
    current_score = get_score(current_set)
    
    def is_valid_add(x):
        # Check if adding x to current_set creates an AP
        # We need to check all pairs in current_set? No, just check if x forms an AP with any pair in current_set.
        # Efficient check:
        # For each a in current_set:
        #   1. Does (a, x, b) exist? b = 2x - a. Check if b in current_set.
        #   2. Does (a, b, x) exist? b = (a+x)/2. Check if b in current_set.
        #   3. Does (x, a, b) exist? b = 2a - x. Check if b in current_set.
        
        for a in current_set:
            if a == x: continue
            
            # Case 1: a, x, b
            b = 2 * x - a
            if 1 <= b <= N and b != x and b in current_set:
                return False
                
            # Case 2: a, b, x
            s = a + x
            if s % 2 == 0:
                b = s // 2
                if b != a and b != x and b in current_set:
                    return False
                    
            # Case 3: x, a, b
            b = 2 * a - x
            if 1 <= b <= N and b != x and b in current_set:
                return False
                
        return True

    def is_valid_remove(x):
        # Removing always keeps it valid
        return True

    # Simulated Annealing Parameters
    T = 10.0
    cooling_rate = 0.999
    min_T = 0.01
    random.seed(42)
    
    best_score = current_score
    best_set = set(current_set)
    
    while T > min_T:
        # Generate a move
        move_type = random.random()
        
        if move_type < 0.5:
            # Try to add a random element
            x = random.randint(1, N)
            if x in current_set:
                continue
            if not is_valid_add(x):
                continue
            
            # Move: Add x
            current_set.add(x)
            in_set[x] = True
            delta_score = 1.0 / x
            new_score = current_score + delta_score
            
            # Acceptance
            if new_score > current_score or random.random() < math.exp(-(-delta_score) / T): # Wait, we maximize.
                # Energy = -Score. Delta E = -Delta Score.
                # Accept if Delta E < 0 (Score increases) or with prob exp(-Delta E / T) = exp(Delta Score / T)
                # So accept if Delta Score > 0 or prob exp(Delta Score / T)
                if delta_score > 0 or random.random() < math.exp(delta_score / T):
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_set = set(current_set)
                else:
                    # Reject
                    current_set.remove(x)
                    in_set[x] = False
            else:
                # This block is redundant if logic above is correct, but let's clean up.
                pass
                
        elif move_type < 0.8:
            # Try to remove a random element
            if not current_set:
                continue
            x = random.choice(list(current_set))
            
            # Move: Remove x
            current_set.remove(x)
            in_set[x] = False
            delta_score = -1.0 / x
            new_score = current_score + delta_score
            
            # Acceptance
            # Delta Score is negative.
            if delta_score > 0 or random.random() < math.exp(delta_score / T):
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_set = set(current_set)
            else:
                # Reject
                current_set.add(x)
                in_set[x] = True
        else:
            # Try swap: remove x, add y
            if not current_set:
                continue
            x = random.choice(list(current_set))
            y = random.randint(1, N)
            if y in current_set:
                continue
            
            # Check if we can remove x and add y
            # Temporarily remove x
            current_set.remove(x)
            in_set[x] = False
            
            if is_valid_add(y):
                current_set.add(y)
                in_set[y] = True
                delta_score = 1.0/y - 1.0/x
                new_score = current_score + delta_score
                
                if delta_score > 0 or random.random() < math.exp(delta_score / T):
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_set = set(current_set)
                else:
                    # Reject
                    current_set.remove(y)
                    in_set[y] = False
                    current_set.add(x)
                    in_set[x] = True
            else:
                # Cannot add y, revert x removal
                current_set.add(x)
                in_set[x] = True
                
        T *= cooling_rate
        
    return list(best_set)
