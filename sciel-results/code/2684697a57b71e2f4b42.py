import time
import random
import math
from collections import deque

N = 12
TIME_LIMIT = 85.0
MAX_SCORE = (N - 1) ** 2

def compute_threshold_bitmask(trans):
    full_mask = (1 << N) - 1
    visited = {full_mask: 0}
    queue = deque([full_mask])
    
    while queue:
        current = queue.popleft()
        dist = visited[current]
        
        if current & (current - 1) == 0:
            return dist
            
        for a in range(2):
            next_set = 0
            temp = current
            while temp:
                lsb = temp & -temp
                s = lsb.bit_length() - 1
                next_set |= 1 << trans[a][s]
                temp -= lsb
                
            if next_set not in visited:
                visited[next_set] = dist + 1
                queue.append(next_set)
                
    return float('inf')

def generate_initial():
    a_map = [(i + 1) % N for i in range(N)]
    b_map = [0 if i == 0 else i - 1 for i in range(N)]
    return [a_map, b_map]

def construct():
    start_time = time.time()
    
    best_trans = generate_initial()
    best_score = compute_threshold_bitmask(best_trans)
    
    if best_score == MAX_SCORE:
        return best_trans

    current_trans = [list(x) for x in best_trans]
    current_score = best_score
    
    T = 10.0
    T_min = 0.1
    cooling = 0.999
    
    while time.time() - start_time < TIME_LIMIT:
        state = random.randint(0, N - 1)
        letter = random.randint(0, 1)
        old_dest = current_trans[letter][state]
        new_dest = random.randint(0, N - 1)
        
        if old_dest == new_dest:
            continue
            
        current_trans[letter][state] = new_dest
        
        new_score = compute_threshold_bitmask(current_trans)
        
        if new_score >= current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_trans = [list(x) for x in current_trans]
                if best_score == MAX_SCORE:
                    break
        else:
            delta = current_score - new_score
            if random.random() < math.exp(-delta / T):
                current_score = new_score
            else:
                current_trans[letter][state] = old_dest
            
        if T > T_min:
            T *= cooling
            if T < T_min:
                T = T_min

    return best_trans
