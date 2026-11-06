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

def _generate_structures():
    """
    Generate a diverse set of structured automata with high inherent synchronization difficulty.
    These serve as high-quality starting points for refinement.
    """
    structures = []
    
    # Type 1: Ping-Pong (Cycle + Collapse)
    # One letter is a cycle, the other maps everything to a few states.
    for target in range(N):
        trans_a = [(i + 1) % N for i in range(N)]
        trans_b = [target] * N
        structures.append([trans_a[:], trans_b[:]])
        
    # Type 2: Ping-Pong with 2-state collapse
    for t1 in range(N):
        for t2 in range(N):
            if t1 != t2:
                trans_a = [(i + 1) % N for i in range(N)]
                trans_b = []
                for i in range(N):
                    if i % 2 == 0:
                        trans_b.append(t1)
                    else:
                        trans_b.append(t2)
                structures.append([trans_a[:], trans_b[:]])
                
    # Type 3: Partial Cycles / Splitting
    # One letter cycles a subset, the other cycles the complement or interacts.
    for split in range(2, N):
        trans_a = []
        trans_b = []
        for i in range(N):
            if i < split:
                trans_a.append((i + 1) % split)
                trans_b.append((i + split) % N if i + split < N else i) # Wrap around to other side? Or just random. Let's do a simple shift.
                # Better: trans_b maps split part to itself or other.
                trans_b.append((i + 1) % N)
            else:
                trans_a.append((i + 1) % N)
                trans_b.append((i - split + 1) % (N - split) + split) # Cycle within the second part
        structures.append([trans_a[:], trans_b[:]])
        
    # Type 4: Randomized Structured (High Entropy but constrained)
    for _ in range(100):
        # Create a base structure: one letter is a permutation (cycle), other is mostly constant
        cycle_len = random.randint(N//2, N)
        perm = list(range(cycle_len))
        random.shuffle(perm)
        # Ensure it's a single cycle or close to it? Just use random perm for now.
        trans_a = []
        for i in range(N):
            if i < cycle_len:
                trans_a.append(perm[i])
            else:
                trans_a.append(random.randint(0, cycle_len - 1))
                
        collapse_target = random.randint(0, N - 1)
        trans_b = []
        for i in range(N):
            if random.random() < 0.8:
                trans_b.append(collapse_target)
            else:
                trans_b.append(random.randint(0, N - 1))
        structures.append([trans_a[:], trans_b[:]])
        
    # Type 5: Known Good Patterns (if any)
    # The "Cherny" automata often have specific properties.
    # Let's add a few manually tuned ones if time permits, but random structured is usually enough.
    
    return structures

def _refine_automaton(trans, deadline):
    """
    Refine an automaton using Simulated Annealing to maximize reset threshold.
    This is the core optimization engine.
    """
    current = [row[:] for row in trans]
    current_thresh = compute_threshold(current)
    
    if current_thresh == -1:
        # If not synchronizing, try to fix it quickly or return None
        # For SA, we usually want to stay in the space of synchronizing automata,
        # but threshold -1 is very low. We can treat it as 0.
        current_thresh = 0
        
    best = [row[:] for row in current]
    best_thresh = current_thresh
    
    T = 10.0
    cooling_rate = 0.995
    min_T = 0.5
    steps_since_improvement = 0
    max_steps_no_improve = 300
    
    while time.time() < deadline:
        # Mutate: change one transition
        new_trans = [row[:] for row in current]
        letter = random.randint(0, 1)
        state = random.randint(0, N - 1)
        new_dest = random.randint(0, N - 1)
        new_trans[letter][state] = new_dest
        
        new_thresh = compute_threshold(new_trans)
        if new_thresh == -1:
            new_thresh = 0
            
        if new_thresh > current_thresh:
            current = new_trans
            current_thresh = new_thresh
            steps_since_improvement = 0
            if current_thresh > best_thresh:
                best_thresh = current_thresh
                best = [row[:] for row in current]
            # Slightly heat up to explore more
            T = min(15.0, T * 1.02)
        else:
            delta = new_thresh - current_thresh
            if delta < 0:
                if T > 0.01:
                    prob = 2.718281828 ** (delta / T)
                else:
                    prob = 0
                if random.random() < prob:
                    current = new_trans
                    current_thresh = new_thresh
            elif delta == 0:
                # Accept moves that keep threshold same to allow exploration
                if random.random() < 0.5:
                    current = new_trans
                    current_thresh = new_thresh
                    
            steps_since_improvement += 1
            T *= cooling_rate
            if T < min_T:
                T = min_T
        
        # Restart if stuck
        if steps_since_improvement > max_steps_no_improve:
            # Perturb current state significantly or restart from a random neighbor
            # Let's just perturb the current best slightly to escape local optima
            for _ in range(5):
                letter = random.randint(0, 1)
                state = random.randint(0, N - 1)
                current[letter][state] = random.randint(0, N - 1)
            current_thresh = compute_threshold(current)
            if current_thresh == -1:
                current_thresh = 0
            steps_since_improvement = 0
            T = 10.0

    return best, best_thresh

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 60s limit
    
    best_trans = None
    best_threshold = 0
    
    # Phase 1: Evaluate structured initial candidates
    structures = _generate_structures()
    
    # Shuffle to avoid bias
    random.shuffle(structures)
    
    for struct in structures:
        if time.time() > deadline - 15:
            break
        thresh = compute_threshold(struct)
        if thresh != -1 and thresh > best_threshold:
            best_threshold = thresh
            best_trans = [row[:] for row in struct]
    
    # Phase 2: Refine the best structured automaton
    if best_trans is not None:
        refined, refined_thresh = _refine_automaton(best_trans, deadline - 5)
        if refined_thresh > best_threshold:
            best_threshold = refined_thresh
            best_trans = refined
    else:
        # Fallback: try a simple synchronizing automaton
        best_trans = [[(i + 1) % N for i in range(N)], [0] * N]
        best_threshold = compute_threshold(best_trans)
        if best_threshold == -1:
            best_threshold = 0

    # Phase 3: Try refining other top candidates or random starts if time permits
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        # Try a few more random starts or other good structures
        candidates = []
        for struct in structures:
            if time.time() > deadline - 5:
                break
            if struct == best_trans:
                continue
            thresh = compute_threshold(struct)
            if thresh != -1:
                candidates.append((thresh, struct))
        
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Refine the next best few
        for thresh, struct in candidates[:3]:
            if time.time() > deadline - 2:
                break
            refined, refined_thresh = _refine_automaton(struct, deadline - 2)
            if refined_thresh > best_threshold:
                best_threshold = refined_thresh
                best_trans = refined

    if best_trans is None:
        # Absolute fallback
        return [[(i + 1) % N for i in range(N)], [0] * N]
        
    return best_trans
