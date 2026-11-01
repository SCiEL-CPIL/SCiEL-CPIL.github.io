"""Greedy set-cover + local search for K_3(6,1)."""
import time
import itertools
import random

# Precompute for the fixed task parameters (q=3, n=6, radius=1)
Q = 3
N = 6
R = 1
SPACE = Q ** N  # 729

def _idx(word):
    v = 0
    for x in word:
        v = v * Q + x
    return v

def _unidx(i):
    w = [0] * N
    for pos in range(N - 1, -1, -1):
        w[pos] = i % Q
        i //= Q
    return w

# Precompute all words and their balls (bit masks over the space)
_ALL_WORDS = [_unidx(i) for i in range(SPACE)]
_BALLS = []
for i in range(SPACE):
    w = _ALL_WORDS[i]
    mask = 1 << i
    for pos in range(N):
        orig = w[pos]
        for d in range(Q):
            if d == orig:
                continue
            w[pos] = d
            mask |= 1 << _idx(w)
        w[pos] = orig
    _BALLS.append(mask)

# Precompute the seed (tetracode direct sum) as a list of indices
def _seed_indices():
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    idxs = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=N - 4):
            idxs.append(_idx(h + list(tail)))
    return idxs

def construct(q, n, radius):
    # Fallback: if parameters differ from the specialized precompute, use a generic greedy.
    if q != Q or n != N or radius != R:
        return _generic_construct(q, n, radius)

    start = time.time()
    deadline = start + 145.0  # safety margin before 150s kill

    # Start from the seed (81 words). Convert to a set of indices.
    chosen = set(_seed_indices())

    # Compute covered mask
    covered = 0
    for c in chosen:
        covered |= _BALLS[c]

    # Greedy: while not fully covered, add the word that covers most uncovered words.
    full_mask = (1 << SPACE) - 1
    while covered != full_mask:
        if time.time() > deadline - 5.0:
            break
        uncovered = full_mask ^ covered
        best_c = -1
        best_gain = 0
        for c in range(SPACE):
            if c in chosen:
                continue
            gain = _BALLS[c] & uncovered
            gc = bin(gain).count('1')
            if gc > best_gain:
                best_gain = gc
                best_c = c
        if best_c < 0:
            break
        chosen.add(best_c)
        covered |= _BALLS[best_c]

    # Local search: try to reduce the number of codewords.
    # Strategy: Repeatedly try to find a pair of codewords (c1, c2) and a new word w
    # such that removing c1 and c2 and adding w still covers everything.
    # This reduces the size by 1.

    def _try_remove_redundant():
        nonlocal chosen, covered
        changed = True
        while changed:
            changed = False
            if time.time() > deadline - 2.0:
                return
            for c in list(chosen):
                others = 0
                for o in chosen:
                    if o == c:
                        continue
                    others |= _BALLS[o]
                if (_BALLS[c] & ~others) == 0:
                    chosen.discard(c)
                    covered = 0
                    for o in chosen:
                        covered |= _BALLS[o]
                    changed = True
                    break

    _try_remove_redundant()

    def _recompute_exclusive():
        excl = {}
        for c in chosen:
            others = 0
            for o in chosen:
                if o == c:
                    continue
                others |= _BALLS[o]
            excl[c] = _BALLS[c] & ~others
        return excl

    improved = True
    while improved:
        improved = False
        if time.time() > deadline - 2.0:
            break
        excl = _recompute_exclusive()
        c_list = list(chosen)
        m = len(c_list)
        found = False
        for i in range(m):
            if found:
                break
            for j in range(i + 1, m):
                if time.time() > deadline - 2.0:
                    found = True
                    break
                need = excl[c_list[i]] | excl[c_list[j]]
                if need == 0:
                    continue
                # Find a word w not in chosen whose ball contains need
                for w in range(SPACE):
                    if w in chosen:
                        continue
                    if (need & ~_BALLS[w]) == 0:
                        chosen.discard(c_list[i])
                        chosen.discard(c_list[j])
                        chosen.add(w)
                        improved = True
                        found = True
                        break
        _try_remove_redundant()

    # Additional heuristic: Try random perturbations if we have time
    if time.time() < deadline - 10.0:
        best_size = len(chosen)
        best_chosen = set(chosen)
        
        # Try to find a better solution by random removals and re-additions
        # This is a simple hill-climbing step
        for _ in range(50): # Limit iterations to keep time low
            if time.time() > deadline - 5.0:
                break
            
            current = set(best_chosen)
            # Randomly remove a word
            to_remove = random.choice(list(current))
            current.discard(to_remove)
            
            # Re-add greedily if necessary
            covered = 0
            for c in current:
                covered |= _BALLS[c]
            
            if covered != full_mask:
                # Greedily add back
                while covered != full_mask:
                    uncovered = full_mask ^ covered
                    best_c = -1
                    best_gain = 0
                    for c in range(SPACE):
                        if c in current:
                            continue
                        gain = _BALLS[c] & uncovered
                        gc = bin(gain).count('1')
                        if gc > best_gain:
                            best_gain = gc
                            best_c = c
                    if best_c < 0:
                        break
                    current.add(best_c)
                    covered |= _BALLS[best_c]
                
                if len(current) < best_size:
                    best_size = len(current)
                    best_chosen = current
                    chosen = current
                    # Run redundancy removal
                    _try_remove_redundant()
                    best_size = len(chosen)
                    best_chosen = set(chosen)

    return [_unidx(c) for c in chosen]

def _generic_construct(q, n, radius):
    """Generic greedy set-cover for arbitrary q, n, radius. Slower but correct."""
    space = q ** n
    words = [list(w) for w in itertools.product(range(q), repeat=n)]

    def hamming(a, b):
        return sum(1 for x, y in zip(a, b) if x != y)

    balls = []
    for i, w in enumerate(words):
        ball = set()
        for j, v in enumerate(words):
            if hamming(w, v) <= radius:
                ball.add(j)
        balls.append(ball)

    full = set(range(space))
    covered = set()
    chosen = []
    chosen_set = set()
    while covered != full:
        best_i = -1
        best_gain = 0
        for i in range(space):
            if i in chosen_set:
                continue
            gain = len(balls[i] - covered)
            if gain > best_gain:
                best_gain = gain
                best_i = i
        if best_i < 0:
            break
        chosen.append(best_i)
        chosen_set.add(best_i)
        covered |= balls[best_i]

    changed = True
    while changed:
        changed = False
        for c in list(chosen):
            others = set()
            for o in chosen:
                if o != c:
                    others |= balls[o]
            if balls[c] <= others:
                chosen.remove(c)
                chosen_set.discard(c)
                covered = set()
                for o in chosen:
                    covered |= balls[o]
                changed = True
                break

    return [words[c] for c in chosen]
