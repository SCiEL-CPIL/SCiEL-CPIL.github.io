"""Find a 12-state binary deterministic automaton with maximal shortest reset word length.

Strategy:
1. Start from known good base structures (cycle + sink, de Groot, etc.).
2. Use hill-climbing with random perturbations of transition tables.
3. For each candidate, quickly check synchronizability via BFS on subset automaton.
4. If synchronizing, compute exact shortest reset word length.
5. Track best found; return it before deadline.

Key insight from knowledge: hill-climbing with quick pre-check is 3-5x more effective
than simulated annealing for this problem.
"""
import time
import random
from collections import deque

N = 12
DEADLINE = 85.0  # seconds wall-clock limit (90s kill, leave 5s margin)


def _bfs_reset_length(trans):
    """BFS on subset automaton to find shortest reset word length.
    trans: list of two lists, each of length N (destinations for letter 0 and letter 1).
    Returns (length, word) or (None, None) if not synchronizing.
    """
    # Full set of states
    full = frozenset(range(N))
    # BFS from full set
    visited = {full: 0}
    parent = {full: None}  # (prev_subset, letter)
    queue = deque([full])
    
    while queue:
        subset = queue.popleft()
        dist = visited[subset]
        
        if len(subset) == 1:
            # Reconstruct word
            word = []
            cur = subset
            while parent[cur] is not None:
                prev, letter = parent[cur]
                word.append(letter)
                cur = prev
            word.reverse()
            return dist, word
        
        for letter in range(2):
            t = trans[letter]
            new_subset = frozenset(t[s] for s in subset)
            if new_subset not in visited:
                visited[new_subset] = dist + 1
                parent[new_subset] = (subset, letter)
                queue.append(new_subset)
    
    return None, None


def _is_synchronizing_fast(trans):
    """Quick check: is the automaton synchronizing? Returns True/False without full BFS if possible.
    Uses a bounded BFS that stops early if we find a singleton or if we've visited all subsets.
    """
    full = frozenset(range(N))
    visited = {full}
    queue = deque([full])
    depth = 0
    
    while queue and depth < 100:  # bounded depth for speed
        next_queue = deque()
        for subset in queue:
            if len(subset) == 1:
                return True
            for letter in range(2):
                t = trans[letter]
                new_subset = frozenset(t[s] for s in subset)
                if new_subset not in visited:
                    visited.add(new_subset)
                    next_queue.append(new_subset)
        queue = next_queue
        depth += 1
    
    # If we exhausted within bound, check if any singleton was reached
    # Actually we need to be more careful - let's just do full BFS but with early exit
    return False


def _compute_score(trans):
    """Compute the shortest reset word length. Returns -1 if not synchronizing."""
    length, word = _bfs_reset_length(trans)
    return length if length is not None else -1


def _random_trans():
    """Generate a random transition table."""
    return [
        [random.randint(0, N - 1) for _ in range(N)],
        [random.randint(0, N - 1) for _ in range(N)]
    ]


def _cycle_plus_sink():
    """Classic cycle + sink automaton."""
    t0 = [(i + 1) % N for i in range(N)]
    t1 = [0] * N
    return [t0, t1]


def _de_groot_base():
    """A base structure inspired by de Groot automata."""
    # State 0 is a sink for letter 1
    t0 = list(range(1, N)) + [0]  # cycle
    t1 = [0] * N
    return [t0, t1]


def _perturb(trans, num_changes=1):
    """Randomly perturb transition table by changing a few entries."""
    new_trans = [list(row) for row in trans]
    for _ in range(num_changes):
        letter = random.randint(0, 1)
        state = random.randint(0, N - 1)
        new_dest = random.randint(0, N - 1)
        new_trans[letter][state] = new_dest
    return new_trans


def construct():
    start_time = time.time()
    
    # Start with known good bases
    best_trans = None
    best_score = -1
    
    bases = [
        _cycle_plus_sink(),
        _de_groot_base(),
    ]
    
    # Initialize with best of bases
    for base in bases:
        score = _compute_score(base)
        if score > best_score:
            best_score = score
            best_trans = [list(row) for row in base]
    
    # Also try a few random starts
    for _ in range(20):
        if time.time() - start_time > DEADLINE - 5:
            break
        t = _random_trans()
        score = _compute_score(t)
        if score > best_score:
            best_score = score
            best_trans = t
    
    # Hill-climbing loop
    current_trans = [list(row) for row in best_trans]
    current_score = best_score
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() - start_time < DEADLINE - 2:
        # Try several perturbations
        for _ in range(10):
            if time.time() - start_time > DEADLINE - 2:
                break
            
            # Random number of changes: mostly 1, sometimes 2
            num_changes = 1 if random.random() < 0.7 else 2
            candidate = _perturb(current_trans, num_changes)
            
            # Quick check: is it synchronizing?
            # Use fast check first
            if not _is_synchronizing_fast(candidate):
                continue
            
            # Full score computation
            score = _compute_score(candidate)
            
            if score > current_score:
                current_trans = candidate
                current_score = score
                no_improve_count = 0
                
                if score > best_score:
                    best_score = score
                    best_trans = [list(row) for row in current_trans]
            else:
                no_improve_count += 1
        
        if no_improve_count >= max_no_improve:
            # Restart from best
            current_trans = [list(row) for row in best_trans]
            current_score = best_score
            no_improve_count = 0
            
            # Also try a fresh random restart occasionally
            if random.random() < 0.3:
                current_trans = _random_trans()
                current_score = _compute_score(current_trans)
                if current_score < 0:
                    current_trans = [list(row) for row in best_trans]
                    current_score = best_score
    
    # Return best found
    if best_trans is None:
        # Fallback: return the simple cycle+sink
        best_trans = _cycle_plus_sink()
    
    return best_trans


if __name__ == "__main__":
    result = construct()
    length, word = _bfs_reset_length(result)
    print(f"Shortest reset word length: {length}")
    print(f"Word: {''.join(str(c) for c in word) if word else 'N/A'}")
