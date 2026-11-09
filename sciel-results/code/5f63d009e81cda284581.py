import time
import random
import math

N = 256
TIME_LIMIT = 45.0  # Safety margin for 90s limit

def construct():
    start_time = time.time()
    
    # Reciprocals
    recip = [0.0] * (N + 1)
    for i in range(1, N + 1):
        recip[i] = 1.0 / i

    # Bitset representation: bit i-1 is 1 if i is in the set
    current_bitset = 0
    current_score = 0.0

    # Helper to check if adding x to bitset creates a 3-AP
    def can_add(bitset, x):
        # x is not in bitset
        # Check for any v in bitset such that {x, v, u} forms a 3-AP for some u in bitset
        # 1. x is middle: u + v = 2x => u = 2x - v
        # 2. v is middle: u + x = 2v => u = 2v - x
        # 3. u is middle: v + x = 2u => u = (v+x)/2
        
        # Iterate through all v in bitset
        # To iterate efficiently, we can just loop 1..N and check bit
        # Since N=256, this is fast enough.
        
        for v in range(1, N + 1):
            if bitset & (1 << (v - 1)):
                # Case 1: x is middle. u = 2x - v
                u = 2 * x - v
                if 1 <= u <= N and u != x: # u != x is always true if x not in set, but u can be x? No, u is in set, x is not.
                    if bitset & (1 << (u - 1)):
                        return False
                
                # Case 2: v is middle. u = 2v - x
                u = 2 * v - x
                if 1 <= u <= N and u != x:
                    if bitset & (1 << (u - 1)):
                        return False
                
                # Case 3: u is middle. v + x = 2u => u = (v+x)/2
                if (v + x) % 2 == 0:
                    u = (v + x) // 2
                    if 1 <= u <= N and u != x and u != v: # u must be distinct from v and x
                        if bitset & (1 << (u - 1)):
                            return False
        return True

    # Helper to get list of elements from bitset
    def get_elements(bitset):
        return [i for i in range(1, N + 1) if bitset & (1 << (i - 1))]

    # Helper to get list of non-elements from bitset
    def get_non_elements(bitset):
        return [i for i in range(1, N + 1) if not (bitset & (1 << (i - 1)))]

    # 1. Greedy Initialization
    # Add elements in order of decreasing reciprocal (increasing value)
    for x in range(1, N + 1):
        if can_add(current_bitset, x):
            current_bitset |= (1 << (x - 1))
            current_score += recip[x]

    best_bitset = current_bitset
    best_score = current_score

    # 2. Simulated Annealing
    temp = 0.05
    cooling_rate = 0.9995
    min_temp = 1e-8
    step_count = 0

    while time.time() - start_time < TIME_LIMIT:
        step_count += 1
        
        if temp < min_temp:
            # Restart
            temp = 0.01
            current_bitset = best_bitset
            current_score = best_score

        move_type = random.random()
        
        if move_type < 0.3:
            # Add move
            non_elements = get_non_elements(current_bitset)
            if non_elements:
                x_add = random.choice(non_elements)
                if can_add(current_bitset, x_add):
                    delta = recip[x_add]
                    # Always accept if valid (delta > 0)
                    current_bitset |= (1 << (x_add - 1))
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_bitset = current_bitset

        elif move_type < 0.5:
            # Remove move
            elements = get_elements(current_bitset)
            if elements:
                x_rem = random.choice(elements)
                delta = -recip[x_rem]
                # Accept with prob exp(delta/temp)
                if random.random() < math.exp(delta / temp):
                    current_bitset &= ~(1 << (x_rem - 1))
                    current_score += delta

        else:
            # Swap move
            elements = get_elements(current_bitset)
            non_elements = get_non_elements(current_bitset)
            if elements and non_elements:
                x_rem = random.choice(elements)
                x_add = random.choice(non_elements)
                
                # Trial: remove x_rem, add x_add
                trial_bitset = current_bitset & ~(1 << (x_rem - 1))
                if can_add(trial_bitset, x_add):
                    trial_bitset |= (1 << (x_add - 1))
                    delta = recip[x_add] - recip[x_rem]
                    
                    if delta > 0:
                        # Always accept improvement
                        current_bitset = trial_bitset
                        current_score += delta
                        if current_score > best_score:
                            best_score = current_score
                            best_bitset = current_bitset
                    else:
                        # Accept worse with prob exp(delta/temp)
                        if random.random() < math.exp(delta / temp):
                            current_bitset = trial_bitset
                            current_score += delta

        temp *= cooling_rate

        # Periodic local optimization: try to add any number that can be added
        if step_count % 500 == 0:
            improved = True
            while improved:
                improved = False
                non_elements = get_non_elements(current_bitset)
                for x in non_elements:
                    if can_add(current_bitset, x):
                        current_bitset |= (1 << (x - 1))
                        current_score += recip[x]
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_bitset = current_bitset

    # Final polish
    improved = True
    while improved:
        improved = False
        non_elements = get_non_elements(best_bitset)
        for x in non_elements:
            if can_add(best_bitset, x):
                best_bitset |= (1 << (x - 1))
                improved = True

    # Convert bitset to list
    result = get_elements(best_bitset)
    
    return result

if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(sum(1.0/x for x in res))
