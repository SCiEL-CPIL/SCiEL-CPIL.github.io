import random
import time
import math
import itertools

def construct(n):
    if n <= 1:
        return "1"
    
    # Generate all permutations
    perms = list(itertools.permutations(range(1, n + 1)))
    N = len(perms)
    
    def get_overlap(a, b):
        k = 0
        for i in range(n - 1):
            if a[n - 1 - i] == b[i]:
                k += 1
            else:
                break
        return k

    def cost(a, b):
        return n - get_overlap(a, b)

    # Initialize state with random permutation of indices
    state = list(range(N))
    random.shuffle(state)
    
    # Calculate initial energy
    energy = 0
    for i in range(N - 1):
        energy += cost(perms[state[i]], perms[state[i + 1]])
    
    best_energy = energy
    best_state = state[:]
    
    start_time = time.time()
    timeout = 110  # 110 seconds to leave buffer
    
    T = 1.0
    cooling = 0.99995
    
    while time.time() - start_time < timeout:
        if T < 1e-6:
            break
            
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        
        if i == j:
            continue
            
        a_idx = state[i]
        b_idx = state[j]
        
        old_energy = 0
        new_energy = 0
        
        if j == i + 1:
            # Adjacent swap
            left_idx = state[i - 1] if i > 0 else None
            right_idx = state[i + 2] if i + 2 < N else None
            
            if i > 0:
                old_energy += cost(perms[left_idx], perms[a_idx])
            if i + 2 < N:
                old_energy += cost(perms[b_idx], perms[right_idx])
            old_energy += cost(perms[a_idx], perms[b_idx])
            
            if i > 0:
                new_energy += cost(perms[left_idx], perms[b_idx])
            if i + 2 < N:
                new_energy += cost(perms[a_idx], perms[right_idx])
            new_energy += cost(perms[b_idx], perms[a_idx])
            
        else:
            # Non-adjacent swap
            left_i = state[i - 1] if i > 0 else None
            right_i = state[i + 1] if i + 1 < N else None
            left_j = state[j - 1] if j > 0 else None
            right_j = state[j + 1] if j + 1 < N else None
            
            if i > 0:
                old_energy += cost(perms[left_i], perms[a_idx])
            if i + 1 < N:
                old_energy += cost(perms[a_idx], perms[right_i])
            if j > 0:
                old_energy += cost(perms[left_j], perms[b_idx])
            if j + 1 < N:
                old_energy += cost(perms[b_idx], perms[right_j])
                
            if i > 0:
                new_energy += cost(perms[left_i], perms[b_idx])
            if i + 1 < N:
                new_energy += cost(perms[b_idx], perms[right_i])
            if j > 0:
                new_energy += cost(perms[left_j], perms[a_idx])
            if j + 1 < N:
                new_energy += cost(perms[a_idx], perms[right_j])
                
        delta = new_energy - old_energy
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            state[i], state[j] = state[j], state[i]
            energy += delta
            if energy < best_energy:
                best_energy = energy
                best_state = state[:]
        
        T *= cooling
        
    # Construct the final string
    final_state = best_state
    word = list(perms[final_state[0]])
    for k in range(1, N):
        prev = perms[final_state[k - 1]]
        curr = perms[final_state[k]]
        ov = get_overlap(prev, curr)
        word.extend(curr[ov:])
        
    return "".join(map(str, word))
