import time
import random
from collections import deque

N = 12

def compute_threshold(trans):
    """
    Compute the shortest reset word length using BFS on the subset automaton.
    trans: list of two lists, each of length N.
    Returns the length of the shortest reset word, or -1 if not synchronizing.
    """
    full_mask = (1 << N) - 1
    start = full_mask
    visited = {start: 0}
    queue = deque([start])
    
    while queue:
        current = queue.popleft()
        dist = visited[current]
        
        if current.bit_count() == 1:
            return dist
            
        # Letter 0
        next_mask = 0
        for i in range(N):
            if (current >> i) & 1:
                next_mask |= 1 << trans[0][i]
        if next_mask not in visited:
            visited[next_mask] = dist + 1
            queue.append(next_mask)
            
        # Letter 1
        next_mask = 0
        for i in range(N):
            if (current >> i) & 1:
                next_mask |= 1 << trans[1][i]
        if next_mask not in visited:
            visited[next_mask] = dist + 1
            queue.append(next_mask)
            
    return -1

def construct():
    start_time = time.time()
    timeout = 55  # Safety margin
    
    best_trans = None
    best_threshold = 0
    
    def random_trans():
        return [
            [random.randint(0, N-1) for _ in range(N)],
            [random.randint(0, N-1) for _ in range(N)]
        ]
        
    def mutate(trans):
        new_trans = [row[:] for row in trans]
        letter = random.randint(0, 1)
        state = random.randint(0, N-1)
        new_dest = random.randint(0, N-1)
        new_trans[letter][state] = new_dest
        return new_trans

    # Initialize with a random automaton
    current_trans = random_trans()
    current_threshold = compute_threshold(current_trans)
    
    if current_threshold > best_threshold:
        best_threshold = current_threshold
        best_trans = [row[:] for row in current_trans]
        
    # Simulated Annealing Parameters
    T = 10.0
    cooling_rate = 0.995
    min_T = 0.5
    steps_since_improvement = 0
    max_steps_no_improve = 500
    
    while time.time() - start_time < timeout:
        new_trans = mutate(current_trans)
        new_threshold = compute_threshold(new_trans)
        
        if new_threshold > current_threshold:
            current_trans = new_trans
            current_threshold = new_threshold
            steps_since_improvement = 0
            if current_threshold > best_threshold:
                best_threshold = current_threshold
                best_trans = [row[:] for row in current_trans]
        else:
            delta = new_threshold - current_threshold
            if delta < 0:
                if T > 0.01:
                    prob = 2.718281828 ** (delta / T)
                else:
                    prob = 0
                if random.random() < prob:
                    current_trans = new_trans
                    current_threshold = new_threshold
                    if current_threshold > best_threshold:
                        best_threshold = current_threshold
                        best_trans = [row[:] for row in current_trans]
            elif delta == 0:
                # Accept moves that keep threshold same to allow exploration
                if random.random() < 0.5:
                    current_trans = new_trans
                    current_threshold = new_threshold
            
            steps_since_improvement += 1
            T *= cooling_rate
            if T < min_T:
                T = min_T
        
        # Restart if stuck
        if steps_since_improvement > max_steps_no_improve:
            current_trans = random_trans()
            current_threshold = compute_threshold(current_trans)
            steps_since_improvement = 0
            T = 10.0
            if current_threshold > best_threshold:
                best_threshold = current_threshold
                best_trans = [row[:] for row in current_trans]

    if best_trans is None:
        # Fallback: known synchronizing automaton
        return [[(i + 1) % N for i in range(N)], [0] * N]
        
    return best_trans
