"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12
import random
import time
from collections import deque

def _bfs_reset_length(trans):
    """Compute the exact shortest reset word length using BFS on subset automaton."""
    full = frozenset(range(N))
    visited = {full: 0}
    queue = deque([full])
    while queue:
        state = queue.popleft()
        dist = visited[state]
        if len(state) == 1:
            return dist
        for letter in range(2):
            new_state = frozenset(trans[letter][s] for s in state)
            if new_state not in visited:
                visited[new_state] = dist + 1
                queue.append(new_state)
    return -1  # Not synchronizing

def _is_synchronizing(trans):
    """Quick check if automaton is synchronizing."""
    return _bfs_reset_length(trans) != -1

def _generate_random_automaton():
    """Generate a random complete binary deterministic automaton."""
    return [
        [random.randint(0, N - 1) for _ in range(N)],
        [random.randint(0, N - 1) for _ in range(N)]
    ]

def _generate_biased_automaton():
    """Generate an automaton biased toward structures with long reset sequences.
    
    Use a structure where one letter is a permutation (cycle) and the other
    has some structure to slowly contract the set.
    """
    # Letter 0: a cycle (permutation)
    perm = list(range(N))
    random.shuffle(perm)
    trans0 = [0] * N
    for i in range(N):
        trans0[i] = perm[i]
    
    # Letter 1: mostly maps to a small subset but with some spread
    # Use a structure where most states map to a few states, but not all the same
    target_states = random.sample(range(N), min(3, N))
    trans1 = []
    for i in range(N):
        if random.random() < 0.7:
            trans1.append(random.choice(target_states))
        else:
            trans1.append(random.randint(0, N - 1))
    
    return [trans0, trans1]

# EVOLVE-BLOCK-START
def construct():
    """Find a synchronizing automaton with maximum reset threshold within time budget."""
    deadline = time.time() + 80  # 80 second deadline (20s safety margin)
    
    best_trans = None
    best_score = -1
    
    # First, try the simple known construction as a baseline
    baseline = [[(i + 1) % N for i in range(N)], [0] * N]
    baseline_score = _bfs_reset_length(baseline)
    if baseline_score > best_score:
        best_score = baseline_score
        best_trans = baseline
    
    # Try biased generation
    for attempt in range(5000):
        if time.time() > deadline:
            break
        
        trans = _generate_biased_automaton()
        score = _bfs_reset_length(trans)
        if score > best_score:
            best_score = score
            best_trans = trans
            # If we hit a very high score, we can stop early
            if best_score >= 100:
                break
    
    # If we haven't found something great, try pure random
    if best_score < 30:
        for attempt in range(20000):
            if time.time() > deadline:
                break
            
            trans = _generate_random_automaton()
            score = _bfs_reset_length(trans)
            if score > best_score:
                best_score = score
                best_trans = trans
    
    # Return the best automaton found
    if best_trans is not None:
        return best_trans
    
    # Fallback: return the simple baseline
    return [[(i + 1) % N for i in range(N)], [0] * N]
# EVOLVE-BLOCK-END
