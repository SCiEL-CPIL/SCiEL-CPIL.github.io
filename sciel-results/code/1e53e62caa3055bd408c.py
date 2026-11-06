"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import deque

N = 12

# EVOLVE-BLOCK-START
def compute_threshold(trans):
    """
    Computes the exact shortest reset word length for a given automaton.
    Returns -1 if the automaton is not synchronizing.
    Uses BFS on the subset automaton.
    """
    # trans is a list of two lists: [trans_a, trans_b]
    # State in BFS is a frozenset of states.
    # Start from the full set of states.
    full_set = frozenset(range(N))
    
    # If it's a singleton, threshold is 0 (already synchronized)
    # But we start from full_set, so we look for the first singleton.
    # Actually, the reset word maps the full set to a singleton.
    # So we BFS from full_set. When we hit a singleton, the distance is the length.
    
    visited = {full_set: 0}
    queue = deque([full_set])
    
    while queue:
        current_set = queue.popleft()
        dist = visited[current_set]
        
        if len(current_set) == 1:
            return dist
        
        # Generate neighbors by applying 'a' and 'b'
        for letter_idx in range(2):
            next_set = frozenset(trans[letter_idx][s] for s in current_set)
            if next_set not in visited:
                visited[next_set] = dist + 1
                queue.append(next_set)
                
    return -1 # Not synchronizing

def is_synchronizing(trans):
    """
    Quick check if an automaton is synchronizing.
    Uses the same BFS but stops early if we find a singleton.
    Returns True if synchronizing, False otherwise.
    """
    full_set = frozenset(range(N))
    visited = {full_set}
    queue = deque([full_set])
    
    while queue:
        current_set = queue.popleft()
        if len(current_set) == 1:
            return True
        
        for letter_idx in range(2):
            next_set = frozenset(trans[letter_idx][s] for s in current_set)
            if next_set not in visited:
                visited.add(next_set)
                queue.append(next_set)
                
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    # Initialize with a known good synchronizing automaton
    # Cycle for a, map to 0 for b
    best_trans = [
        [(i + 1) % N for i in range(N)],
        [0] * N
    ]
    best_score = compute_threshold(best_trans)
    
    # Random seed for reproducibility in local runs, but evaluator might run multiple times
    random.seed(42)
    
    while time.time() < deadline:
        # Generate a candidate
        # Start from current best or a random one? 
        # Hill climbing from current best is usually better for local optima.
        # But we might get stuck. Let's do a mix: 90% mutate best, 10% restart random.
        
        if random.random() < 0.1:
            # Random restart
            candidate = [
                [random.randint(0, N-1) for _ in range(N)],
                [random.randint(0, N-1) for _ in range(N)]
            ]
        else:
            # Mutate current best
            candidate = [list(t) for t in best_trans]
            # Pick a random transition to change
            letter = random.randint(0, 1)
            state = random.randint(0, N-1)
            candidate[letter][state] = random.randint(0, N-1)
        
        # Check if synchronizing
        if not is_synchronizing(candidate):
            continue
        
        # Compute exact threshold
        score = compute_threshold(candidate)
        
        if score > best_score:
            best_score = score
            best_trans = candidate
            
            # Optional: If we find a very high score, we could try to refine it further
            # but for now, just keep updating best.
            
    return best_trans
# EVOLVE-BLOCK-END
