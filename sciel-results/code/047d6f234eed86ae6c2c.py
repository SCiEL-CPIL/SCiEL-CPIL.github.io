import time
import random
from fractions import Fraction

def construct():
    N = 256
    start_time = time.time()
    time_limit = 50.0  # Safety margin
    
    # Helper to check if x can be added to set S without creating a 3-AP
    def can_add(x, S):
        if x in S:
            return False
        # Check if x forms a 3-AP with any two elements in S
        # A 3-AP is (a, b, c) such that a+c = 2b.
        # If we add x, it can be a, b, or c.
        # 1. x is middle: x = (a+c)/2 => 2x = a+c. 
        #    Check if any pair in S sums to 2x.
        #    This is O(K^2) if done naively. 
        #    Alternative: For each a in S, let c = 2x - a. If c in S and c != a, then (a, x, c) is AP.
        #    This is O(K).
        # 2. x is end: 
        #    a, b, x => b = (a+x)/2 => 2b = a+x.
        #    For each b in S, let a = 2b - x. If a in S and a != b and a != x, then (a, b, x) is AP.
        #    This is O(K).
        #    x, b, c => b = (x+c)/2 => 2b = x+c.
        #    For each b in S, let c = 2b - x. If c in S and c != b and c != x, then (x, b, c) is AP.
        #    This is O(K).
        
        # Combining these:
        # If we iterate a in S:
        #   c1 = 2*x - a. If c1 in S and c1 != a: AP (a, x, c1)
        #   c2 = 2*a - x. If c2 in S and c2 != a: AP (c2, a, x) -- wait, if c2 in S, then (c2, a, x) is AP?
        #       c2 = 2a - x => a = (c2+x)/2. Yes.
        #       Note: c1 and c2 are symmetric in some sense.
        #       Actually, checking "is there a in S such that 2x-a in S" covers the case where x is the middle.
        #       Checking "is there a in S such that 2a-x in S" covers the case where x is an end?
        #       Let's verify.
        #       AP: (u, v, w). 
        #       If x=u: v=(x+w)/2 => w=2v-x. Check if for any v in S, 2v-x in S.
        #       If x=v: u+w=2x => w=2x-u. Check if for any u in S, 2x-u in S.
        #       If x=w: u+v=2w? No, v=(u+w)/2 => u=2v-w=2v-x. Check if for any v in S, 2v-x in S.
        
        # So we need to check:
        # 1. For any a in S, is (2*x - a) in S? (x is middle)
        # 2. For any a in S, is (2*a - x) in S? (x is end, a is middle)
        
        # Note: If (2*x - a) in S, let b = 2*x - a. Then a+x = x+b => x is middle of a,b? No, a+b = 2x. So x is middle.
        # If (2*a - x) in S, let b = 2*a - x. Then b+x = 2a. So a is middle of b,x.
        
        # So we just iterate through S once and check both conditions?
        # Actually, condition 2 for 'a' is condition 1 for 'a' if we swap roles? No.
        # Let's just check both.
        
        for a in S:
            # x is middle
            b = 2 * x - a
            if b in S and b != a:
                return False
            # x is end (a is middle)
            b = 2 * a - x
            if b in S and b != a:
                return False
        return True

    def get_score(S):
        return sum(1.0 / x for x in S)

    # 1. Greedy Initialization
    S_greedy = set()
    for x in range(1, N + 1):
        if can_add(x, S_greedy):
            S_greedy.add(x)
    
    current_S = set(S_greedy)
    current_score = get_score(current_S)
    best_S = set(current_S)
    best_score = current_score
    
    all_numbers = list(range(1, N + 1))
    
    # 2. Iterated Local Search
    while time.time() - start_time < time_limit:
        # Random Move
        move_type = random.choice(['add', 'remove', 'swap'])
        
        new_S = set(current_S)
        valid = True
        
        if move_type == 'add':
            candidates = [x for x in all_numbers if x not in current_S]
            if not candidates: continue
            x = random.choice(candidates)
            if can_add(x, new_S):
                new_S.add(x)
            else:
                continue # Invalid move, skip
        elif move_type == 'remove':
            if not current_S: continue
            x = random.choice(list(current_S))
            new_S.remove(x)
        elif move_type == 'swap':
            if not current_S: continue
            x = random.choice(list(current_S))
            candidates = [y for y in all_numbers if y not in current_S]
            if not candidates: continue
            y = random.choice(candidates)
            new_S.remove(x)
            if can_add(y, new_S):
                new_S.add(y)
            else:
                continue # Invalid swap
        
        new_score = get_score(new_S)
        
        # Hill Climbing: Accept if better
        if new_score > current_score:
            current_S = new_S
            current_score = new_score
            if current_score > best_score:
                best_S = set(current_S)
                best_score = current_score
        else:
            # Simple stochastic acceptance or restart logic?
            # Let's add a small probability of accepting worse moves to escape local optima
            # Or just count steps and restart.
            # Given time budget, let's try to escape by perturbation if stuck.
            pass

    # Return best set found
    return list(best_S)
