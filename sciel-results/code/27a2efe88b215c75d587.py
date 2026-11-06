import time
import random
import math
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

def _refine_automaton(trans, deadline, max_iters=5000):
    """
    Refine an automaton using simulated annealing to maximize reset threshold.
    """
    current = [row[:] for row in trans]
    current_thresh = compute_threshold(current)
    best = [row[:] for row in current]
    best_thresh = current_thresh
    
    T = 5.0
    cooling_rate = 0.995
    min_T = 0.1
    iter_count = 0
    
    while time.time() < deadline and T > min_T and iter_count < max_iters:
        # Mutate: change one transition
        new_trans = [row[:] for row in current]
        letter = random.randint(0, 1)
        state = random.randint(0, N - 1)
        new_dest = random.randint(0, N - 1)
        new_trans[letter][state] = new_dest
        
        new_thresh = compute_threshold(new_trans)
        
        if new_thresh > current_thresh:
            current = new_trans
            current_thresh = new_thresh
            if current_thresh > best_thresh:
                best = [row[:] for row in current]
                best_thresh = current_thresh
            T = max(min_T, T * 1.05)  # Heat up slightly
        else:
            delta = new_thresh - current_thresh
            if delta < 0:
                prob = math.exp(delta / T) if T > 0.01 else 0
                if random.random() < prob:
                    current = new_trans
                    current_thresh = new_thresh
            elif delta == 0:
                if random.random() < 0.3:
                    current = new_trans
            
            T *= cooling_rate
            if T < min_T:
                T = min_T
        
        iter_count += 1
    
    return best, best_thresh

def _beam_search_fill(core_trans, deadline, beam_width=10):
    """
    Hierarchical decomposition: fix a high-threshold core and optimize
    remaining entries via targeted beam search.
    """
    # Start with the core
    current = [row[:] for row in core_trans]
    current_thresh = compute_threshold(current)
    best = [row[:] for row in current]
    best_thresh = current_thresh
    
    # Identify which transitions are "flexible" - we'll treat all as flexible
    # but use beam search to explore promising modifications
    beam = [(current, current_thresh)]
    
    while time.time() < deadline:
        next_beam = []
        
        for trans, thresh in beam:
            # Generate neighbors by changing one transition
            for letter in range(2):
                for state in range(N):
                    for new_dest in range(N):
                        if trans[letter][state] == new_dest:
                            continue
                        
                        new_trans = [row[:] for row in trans]
                        new_trans[letter][state] = new_dest
                        new_thresh = compute_threshold(new_trans)
                        
                        if new_thresh > -1:  # Only keep synchronizing automata
                            next_beam.append((new_trans, new_thresh))
        
        if not next_beam:
            break
        
        # Sort by threshold and keep top beam_width
        next_beam.sort(key=lambda x: x[1], reverse=True)
        beam = next_beam[:beam_width]
        
        # Update best
        if beam[0][1] > best_thresh:
            best_thresh = beam[0][1]
            best = [row[:] for row in beam[0][0]]
        
        # If we've reached a high threshold, stop
        if best_thresh >= 60:
            break
    
    return best, best_thresh

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 60s limit
    
    best_trans = None
    best_threshold = 0
    
    # Phase 1: Try structured ping-pong automata
    # Type 1: Cycle + collapse to single state
    for start_state in range(N):
        if time.time() > deadline - 20:
            break
        trans_a = [(i + 1) % N for i in range(N)]
        trans_b = [start_state] * N
        thresh = compute_threshold([trans_a, trans_b])
        if thresh > best_threshold:
            best_threshold = thresh
            best_trans = [trans_a[:], trans_b[:]]
    
    # Type 2: Cycle + collapse to two states
    for start_state in range(N):
        for second_state in range(N):
            if second_state == start_state:
                continue
            if time.time() > deadline - 15:
                break
            trans_a = [(i + 1) % N for i in range(N)]
            trans_b = []
            for i in range(N):
                if i % 2 == 0:
                    trans_b.append(start_state)
                else:
                    trans_b.append(second_state)
            thresh = compute_threshold([trans_a, trans_b])
            if thresh > best_threshold:
                best_threshold = thresh
                best_trans = [trans_a[:], trans_b[:]]
    
    # Phase 2: Use beam search to refine from the best structured automaton
    if best_trans is not None:
        refined, refined_thresh = _beam_search_fill(best_trans, deadline - 10, beam_width=5)
        if refined_thresh > best_threshold:
            best_threshold = refined_thresh
            best_trans = refined
    
    # Phase 3: Simulated annealing polish
    if best_trans is not None and time.time() < deadline - 5:
        refined, refined_thresh = _refine_automaton(best_trans, deadline - 2)
        if refined_thresh > best_threshold:
            best_threshold = refined_thresh
            best_trans = refined
    
    # Fallback
    if best_trans is None:
        best_trans = [[(i + 1) % N for i in range(N)], [0] * N]
        best_threshold = compute_threshold(best_trans)
    
    return best_trans
