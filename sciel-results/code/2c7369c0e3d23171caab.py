import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit

    points = list(range(16))
    all_triples = list(combinations(points, 3))
    N = len(all_triples)

    # Precompute masks and pair indices
    triple_mask = [0] * N
    triple_pairs = [[] for _ in range(N)]
    for i, t in enumerate(all_triples):
        triple_mask[i] = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_pairs[i] = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]

    # Pair -> set of triple indices containing that pair
    pair_to_triples = {}
    for i, t in enumerate(all_triples):
        for p in triple_pairs[i]:
            pair_to_triples.setdefault(p, []).append(i)

    # Precompute subset masks for alpha computation
    # We need to check if a subset of points contains any triple
    # For a given set of active triples (by mask), check if subset is independent

    def compute_alpha(active_indices):
        """Compute independence number for a set of active triple indices."""
        if not active_indices:
            return 16
        # Build a list of masks for active triples
        masks = [triple_mask[i] for i in active_indices]
        
        # Check from 16 down to 1
        for k in range(16, 0, -1):
            for subset in combinations(points, k):
                smask = 0
                for p in subset:
                    smask |= (1 << p)
                is_ind = True
                for m in masks:
                    if (m & smask) == m:
                        is_ind = False
                        break
                if is_ind:
                    return k
        return 0

    def find_independent_set(active_indices, k):
        """Find an independent set of size k, or None."""
        if k > 16:
            return None
        if not active_indices:
            return set(points[:k])
        masks = [triple_mask[i] for i in active_indices]
        for subset in combinations(points, k):
            smask = 0
            for p in subset:
                smask |= (1 << p)
            is_ind = True
            for m in masks:
                if (m & smask) == m:
                    is_ind = False
                    break
            if is_ind:
                return set(subset)
        return None

    def is_compatible(idx, used_pairs):
        """Check if triple idx can be added given used pairs."""
        for p in triple_pairs[idx]:
            if p in used_pairs:
                return False
        return True

    def build_greedy_sparse():
        """Build a random sparse packing."""
        indices = list(range(N))
        random.shuffle(indices)
        current = []
        used_pairs = set()
        # Only add a fraction of possible triples to keep it sparse
        for idx in indices:
            if is_compatible(idx, used_pairs):
                current.append(idx)
                for p in triple_pairs[idx]:
                    used_pairs.add(p)
        return current, used_pairs

    # Strategy: Start with various sparse packings, then optimize
    best_alpha = 16
    best_indices = []

    # Phase 1: Try multiple random sparse starts
    init_deadline = start_time + 10
    attempts = 0
    while time.time() < init_deadline and attempts < 50:
        attempts += 1
        # Random subset of triples to start with
        num_to_try = random.randint(5, 25)
        random_indices = random.sample(range(N), min(num_to_try, N))
        random.shuffle(random_indices)
        
        current = []
        used_pairs = set()
        for idx in random_indices:
            if is_compatible(idx, used_pairs):
                current.append(idx)
                for p in triple_pairs[idx]:
                    used_pairs.add(p)
        
        if not current:
            continue
        
        alpha = compute_alpha(current)
        if alpha < best_alpha:
            best_alpha = alpha
            best_indices = list(current)
            if best_alpha <= 6:
                break

    # Also try a few greedy maximal starts
    for _ in range(5):
        if time.time() >= init_deadline:
            break
        current, used_pairs = build_greedy_sparse()
        alpha = compute_alpha(current)
        if alpha < best_alpha:
            best_alpha = alpha
            best_indices = list(current)

    if not best_indices:
        # Fallback: just return empty or a minimal set
        return [[0, 1, 2]]

    # Phase 2: Local search - add, remove, swap triples
    current_indices = list(best_indices)
    current_alpha = best_alpha
    no_improve = 0

    # Build used_pairs from current
    def rebuild_used_pairs(indices):
        up = set()
        for idx in indices:
            for p in triple_pairs[idx]:
                up.add(p)
        return up

    used_pairs = rebuild_used_pairs(current_indices)

    while time.time() < deadline:
        if no_improve > 200:
            # Restart with a new random perturbation
            # Remove some random triples
            if current_indices:
                for _ in range(random.randint(1, min(5, len(current_indices)))):
                    if current_indices:
                        rm = random.choice(current_indices)
                        current_indices.remove(rm)
                        for p in triple_pairs[rm]:
                            used_pairs.discard(p)
            no_improve = 0
            continue

        # Compute current alpha
        current_alpha = compute_alpha(current_indices)
        if current_alpha < best_alpha:
            best_alpha = current_alpha
            best_indices = list(current_indices)
            no_improve = 0
        else:
            no_improve += 1

        if best_alpha <= 5:
            # Very good, keep trying for 4
            pass

        # Get an independent set to target
        target_k = current_alpha
        S = find_independent_set(current_indices, target_k)

        if S is None:
            continue

        # Generate candidates: triples entirely within S
        candidates = []
        S_set = set(S)
        for i, t in enumerate(all_triples):
            if t[0] in S_set and t[1] in S_set and t[2] in S_set:
                candidates.append(i)

        random.shuffle(candidates)

        # Try to add a candidate
        added = False
        for idx in candidates:
            if is_compatible(idx, used_pairs):
                current_indices.append(idx)
                for p in triple_pairs[idx]:
                    used_pairs.add(p)
                added = True
                break

        if not added:
            # Try a swap: remove one existing triple, add a candidate
            # Find blocking pairs
            blocking_pairs = set()
            for idx in candidates:
                for p in triple_pairs[idx]:
                    if p in used_pairs:
                        blocking_pairs.add(p)

            # Find triples that use blocking pairs
            movable = []
            for idx in current_indices:
                for p in triple_pairs[idx]:
                    if p in blocking_pairs:
                        movable.append(idx)
                        break

            if not movable:
                movable = list(current_indices)

            if movable:
                old_idx = random.choice(movable)
                old_pairs = triple_pairs[old_idx]
                # Remove old
                current_indices.remove(old_idx)
                for p in old_pairs:
                    used_pairs.discard(p)

                # Try to add a candidate
                swap_ok = False
                for idx in candidates:
                    if is_compatible(idx, used_pairs):
                        current_indices.append(idx)
                        for p in triple_pairs[idx]:
                            used_pairs.add(p)
                        swap_ok = True
                        break

                if not swap_ok:
                    # Revert
                    current_indices.append(old_idx)
                    for p in old_pairs:
                        used_pairs.add(p)

        # Also try pure removal: remove a random triple to see if alpha drops
        # (removal can help if the triple was "blocking" in a bad way)
        if current_indices and random.random() < 0.1:
            rm_idx = random.choice(current_indices)
            rm_pairs = triple_pairs[rm_idx]
            current_indices.remove(rm_idx)
            for p in rm_pairs:
                used_pairs.discard(p)

    # Convert best_indices back to triples
    result = [list(all_triples[i]) for i in best_indices]
    return result
