"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import deque

N = 12

# EVOLVE-BLOCK-START

def _bfs_threshold(trans):
    """
    Compute the exact shortest reset word length via BFS on the subset automaton.
    Returns (threshold, is_synchronizing).
    """
    full = tuple(range(N))
    visited = {full}
    queue = deque()
    queue.append(full)
    depth = 0
    while queue:
        size = len(queue)
        for _ in range(size):
            subset = queue.popleft()
            if len(subset) == 1:
                return depth, True
            for a in range(2):
                img = frozenset(trans[a][s] for s in subset)
                key = frozenset(img)  # use frozenset for hashing
                if key not in visited:
                    visited.add(key)
                    queue.append(tuple(sorted(img)))
        depth += 1
    return None, False


def _generate_pawlik():
    """Generate the standard Pawlik (Černý) automaton for N states."""
    sigma0 = [(i + 1) % N for i in range(N)]
    sigma1 = [0 if i == 0 else i - 1 for i in range(N)]
    return [sigma0, sigma1]


def _generate_pawlik_variant(rng):
    """
    Generate a variant of the Pawlik automaton by slightly perturbing the transitions.
    This explores the neighborhood of the extremal construction.
    """
    trans = _generate_pawlik()
    # Perturb a few transitions in sigma1
    for _ in range(rng.randint(1, 4)):
        i = rng.randint(1, N - 1)
        new_dest = rng.randint(0, N - 1)
        trans[1][i] = new_dest
    # Optionally perturb sigma0 slightly
    if rng.random() < 0.3:
        i = rng.randint(0, N - 1)
        new_dest = (i + 1 + rng.randint(-1, 1)) % N
        if new_dest != (i + 1) % N:
            trans[0][i] = new_dest
    return trans


def _generate_random_synchronizing(rng, max_attempts=2000):
    """
    Generate random automata and check if they synchronize.
    Return the one with the highest threshold found.
    """
    best_trans = None
    best_thresh = -1
    
    for _ in range(max_attempts):
        trans = [
            [rng.randint(0, N - 1) for _ in range(N)],
            [rng.randint(0, N - 1) for _ in range(N)]
        ]
        # Quick check: ensure it's not trivially non-synchronizing
        # Compute threshold
        thresh, sync = _bfs_threshold(trans)
        if sync and thresh > best_thresh:
            best_thresh = thresh
            best_trans = trans
            if best_thresh >= 121:
                break
    return best_trans


def _local_search(trans, rng, max_steps=500):
    """
    Perform local search to improve the threshold.
    Mutate one transition at a time and keep improvements.
    """
    current = [row[:] for row in trans]
    current_thresh, _ = _bfs_threshold(current)
    
    if current_thresh is None:
        return None
    
    for _ in range(max_steps):
        # Pick a random position to mutate
        letter = rng.randint(0, 1)
        state = rng.randint(0, N - 1)
        old_dest = current[letter][state]
        new_dest = rng.randint(0, N - 1)
        if new_dest == old_dest:
            continue
        
        current[letter][state] = new_dest
        new_thresh, sync = _bfs_threshold(current)
        
        if sync and (new_thresh is not None and new_thresh > current_thresh):
            current_thresh = new_thresh
        else:
            # Revert
            current[letter][state] = old_dest
    
    return current


def construct():
    deadline = time.time() + 80  # Safety margin before 90s kill
    
    # Start with the Pawlik automaton as baseline
    best_trans = _generate_pawlik()
    best_thresh, _ = _bfs_threshold(best_trans)
    
    if best_thresh is None:
        # Should not happen for Pawlik, but fallback
        best_trans = [[(i + 1) % N for i in range(N)], [0] * N]
        best_thresh = 1
    
    rng = random.Random(42)
    
    # Try local search from Pawlik
    if time.time() < deadline - 10:
        improved = _local_search(best_trans, rng, max_steps=300)
        if improved is not None:
            imp_thresh, _ = _bfs_threshold(improved)
            if imp_thresh is not None and imp_thresh > best_thresh:
                best_trans = improved
                best_thresh = imp_thresh
    
    # Try Pawlik variants
    if time.time() < deadline - 10:
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            variant = _generate_pawlik_variant(rng)
            thresh, sync = _bfs_threshold(variant)
            if sync and thresh > best_thresh:
                best_thresh = thresh
                best_trans = variant
                # Try local search from this variant
                if time.time() < deadline - 10:
                    improved = _local_search(best_trans, rng, max_steps=100)
                    if improved is not None:
                        imp_thresh, _ = _bfs_threshold(improved)
                        if imp_thresh is not None and imp_thresh > best_thresh:
                            best_thresh = imp_thresh
                            best_trans = improved
    
    # Try random automata with local search
    if time.time() < deadline - 15:
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            rand_trans = _generate_random_synchronizing(rng, max_attempts=100)
            if rand_trans is not None:
                thresh, _ = _bfs_threshold(rand_trans)
                if thresh > best_thresh:
                    best_thresh = thresh
                    best_trans = rand_trans
                    # Local search
                    if time.time() < deadline - 10:
                        improved = _local_search(best_trans, rng, max_steps=100)
                        if improved is not None:
                            imp_thresh, _ = _bfs_threshold(improved)
                            if imp_thresh is not None and imp_thresh > best_thresh:
                                best_thresh = imp_thresh
                                best_trans = improved
    
    return best_trans

# EVOLVE-BLOCK-END
