N = 16
import time
import random

def _build_initial_masks():
    masks = []
    for i in range(N):
        m = 0
        for k in range(1 << N):
            if (k >> i) & 1:
                m |= (1 << k)
        masks.append(m)
    return masks

def _build_target_masks():
    targets = []
    for i in range(N):
        m = 0
        for k in range(1 << N):
            if bin(k).count('1') >= N - i:
                m |= (1 << k)
        targets.append(m)
    return targets

_initial = _build_initial_masks()
_targets = _build_target_masks()

def _odd_even_network():
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _apply(comparators, init):
    masks = list(init)
    for i, j in comparators:
        a, b = masks[i], masks[j]
        masks[i] = a & b
        masks[j] = a | b
    return masks

def _valid(comparators):
    final = _apply(comparators, _initial)
    return all(final[i] == _targets[i] for i in range(N))

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 78  # safety margin
    
    current = _odd_even_network()
    best = list(current)
    best_len = len(current)
    
    assert _valid(best), "Initial network must be valid"
    
    rng = random.Random(12345)
    
    while time.time() < deadline:
        improved = False
        
        # Deletion pass
        for idx in range(len(current) - 1, -1, -1):
            if time.time() >= deadline:
                break
            cand = current[:idx] + current[idx+1:]
            if _valid(cand):
                current = cand
                improved = True
                if len(current) < best_len:
                    best = list(current)
                    best_len = len(current)
                break
        
        if time.time() >= deadline:
            break
        
        # Swap pass
        if not improved:
            for idx in range(len(current) - 1):
                if time.time() >= deadline:
                    break
                cand = current[:]
                cand[idx], cand[idx+1] = current[idx+1], current[idx]
                if _valid(cand):
                    current = cand
                    improved = True
                    break
        
        if time.time() >= deadline:
            break
        
        # Insertion
        if not improved:
            i = rng.randint(0, N - 2)
            j = rng.randint(i + 1, N - 1)
            pos = rng.randint(0, len(current))
            cand = current[:pos] + [(i, j)] + current[pos:]
            if _valid(cand):
                current = cand
                if len(current) < best_len:
                    best = list(current)
                    best_len = len(current)
        
        if not improved:
            # Perturb
            if best_len < len(current):
                current = list(best)
            for _ in range(rng.randint(2, 5)):
                idx = rng.randint(0, len(current) - 1)
                current = current[:idx] + current[idx+1:]
            if not _valid(current):
                current = list(best)
    
    return best
# EVOLVE-BLOCK-END
