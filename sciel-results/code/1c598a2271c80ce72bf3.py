import time
import random
import sys

N = 9
SIZE = 1 << N
MASK = SIZE - 1

# Precompute neighbor masks
NEIGHBOR_MASKS = [0] * SIZE
for v in range(SIZE):
    m = 0
    for b in range(N):
        m |= 1 << (v ^ (1 << b))
    NEIGHBOR_MASKS[v] = m

def construct():
    deadline = time.time() + 113.0
    best_path = [0]
    best_len = 1
    rng = random.Random(12345)

    def greedy_extend(path, used_mask, forbidden_mask, strategy):
        """Extend a path greedily using a given strategy."""
        cur = path[-1]
        while True:
            candidates = NEIGHBOR_MASKS[cur] & ~(used_mask | forbidden_mask)
            if candidates == 0:
                break
            # Choose candidate based on strategy
            tmp = candidates
            best_v = -1
            if strategy == 'min_degree':
                # Pick neighbor with fewest available options (most constrained)
                min_opts = SIZE + 1
                while tmp:
                    lsb = tmp & (-tmp)
                    v = lsb.bit_length() - 1
                    tmp ^= lsb
                    # Count how many of v's neighbors are still available
                    avail = NEIGHBOR_MASKS[v] & ~(used_mask | (1 << v) | forbidden_mask)
                    cnt = bin(avail).count('1')
                    if cnt < min_opts:
                        min_opts = cnt
                        best_v = v
            elif strategy == 'max_degree':
                max_opts = -1
                while tmp:
                    lsb = tmp & (-tmp)
                    v = lsb.bit_length() - 1
                    tmp ^= lsb
                    avail = NEIGHBOR_MASKS[v] & ~(used_mask | (1 << v) | forbidden_mask)
                    cnt = bin(avail).count('1')
                    if cnt > max_opts:
                        max_opts = cnt
                        best_v = v
            elif strategy == 'random':
                # Pick random candidate
                bits = []
                tmp2 = candidates
                while tmp2:
                    lsb = tmp2 & (-tmp2)
                    bits.append(lsb.bit_length() - 1)
                    tmp2 ^= lsb
                best_v = rng.choice(bits)
            else:  # 'first'
                lsb = candidates & (-candidates)
                best_v = lsb.bit_length() - 1

            if best_v < 0:
                break

            path.append(best_v)
            used_mask |= (1 << best_v)
            forbidden_mask |= NEIGHBOR_MASKS[cur]
            cur = best_v

        return path, used_mask, forbidden_mask

    def beam_search(start, width, max_steps):
        """Beam search with bitmask state."""
        # State: (path, used_mask, forbidden_mask)
        # forbidden_mask: neighbors of all non-last vertices in path
        beam = [([start], 1 << start, 0)]
        best_local = [start]
        best_local_len = 1

        for step in range(max_steps):
            if time.time() > deadline - 3:
                break

            new_beam = []
            seen_keys = set()

            for path, used_mask, forbidden_mask in beam:
                cur = path[-1]
                candidates = NEIGHBOR_MASKS[cur] & ~(used_mask | forbidden_mask)

                if candidates == 0:
                    if len(path) > best_local_len:
                        best_local_len = len(path)
                        best_local = list(path)
                    continue

                # Expand all candidates
                tmp = candidates
                while tmp:
                    lsb = tmp & (-tmp)
                    v = lsb.bit_length() - 1
                    tmp ^= lsb

                    new_path = path + [v]
                    new_used = used_mask | (1 << v)
                    new_forbidden = forbidden_mask | NEIGHBOR_MASKS[cur]

                    # Diversity key: last 4 vertices
                    key = (new_path[-4], new_path[-3] if len(new_path) >= 3 else -1,
                           new_path[-2] if len(new_path) >= 2 else -1, v)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)

                    new_beam.append((new_path, new_used, new_forbidden))

                    if len(new_path) > best_local_len:
                        best_local_len = len(new_path)
                        best_local = list(new_path)

            if not new_beam:
                break

            # Prune: sort by length descending, keep top width
            new_beam.sort(key=lambda x: -len(x[0]))
            beam = new_beam[:width]

        return best_local, best_local_len

    # Phase 1: Quick greedy constructions with different strategies
    strategies = ['min_degree', 'max_degree', 'random', 'first']
    for s in range(20):
        if time.time() > deadline - 50:
            break
        start = rng.randint(0, SIZE - 1)
        strat = strategies[s % len(strategies)]
        path, used, forb = greedy_extend([start], 1 << start, 0, strat)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path

    # Phase 2: Beam searches with varying widths
    widths = [20, 50, 100, 200, 500]
    max_steps_list = [80, 70, 60, 50, 40]

    while time.time() < deadline - 15:
        w = rng.choice(widths)
        idx = widths.index(w)
        ms = max_steps_list[idx]
        start = rng.randint(0, SIZE - 1)

        p, L = beam_search(start, w, ms)
        if L > best_len:
            best_len = L
            best_path = p

        # Try to extend the best path found so far
        if time.time() < deadline - 10:
            cur_best = best_path
            used_mask = 0
            for v in cur_best:
                used_mask |= (1 << v)
            # Build forbidden mask from all non-last vertices
            forbidden_mask = 0
            for i in range(len(cur_best) - 1):
                forbidden_mask |= NEIGHBOR_MASKS[cur_best[i]]

            # Try extending with different strategies
            for strat in ['min_degree', 'max_degree', 'random']:
                trial = list(cur_best)
                t_used = used_mask
                t_forb = forbidden_mask
                trial, t_used, t_forb = greedy_extend(trial, t_used, t_forb, strat)
                if len(trial) > best_len:
                    best_len = len(trial)
                    best_path = trial
                    # Update masks for further extension
                    used_mask = t_used
                    forbidden_mask = t_forb

    # Phase 3: Final extension attempts on the best path
    if time.time() < deadline - 2:
        cur_best = best_path
        used_mask = 0
        for v in cur_best:
            used_mask |= (1 << v)
        forbidden_mask = 0
        for i in range(len(cur_best) - 1):
            forbidden_mask |= NEIGHBOR_MASKS[cur_best[i]]

        for _ in range(50):
            trial = list(cur_best)
            t_used = used_mask
            t_forb = forbidden_mask
            strat = rng.choice(['min_degree', 'max_degree', 'random', 'first'])
            trial, t_used, t_forb = greedy_extend(trial, t_used, t_forb, strat)
            if len(trial) > best_len:
                best_len = len(trial)
                best_path = trial
                used_mask = t_used
                forbidden_mask = t_forb
            else:
                # Try a small perturbation: remove last vertex, try different extension
                if len(trial) > 2:
                    perturbed = cur_best[:-1]
                    p_used = 0
                    for v in perturbed:
                        p_used |= (1 << v)
                    p_forb = 0
                    for i in range(len(perturbed) - 1):
                        p_forb |= NEIGHBOR_MASKS[perturbed[i]]
                    perturbed, p_used, p_forb = greedy_extend(perturbed, p_used, p_forb, strat)
                    if len(perturbed) > best_len:
                        best_len = len(perturbed)
                        best_path = perturbed
                        used_mask = p_used
                        forbidden_mask = p_forb

    return best_path


if __name__ == "__main__":
    res = construct()
    print(f"Path length: {len(res)}")
    # Quick validation
    if len(res) > 1:
        valid = True
        seen = set()
        for i, v in enumerate(res):
            if v in seen:
                valid = False
                break
            seen.add(v)
            if i > 0:
                diff = v ^ res[i-1]
                if diff & (diff - 1) != 0:  # not a power of 2
                    valid = False
                    break
        if valid:
            # Check non-consecutive pairs differ by at least 2 bits
            for i in range(len(res)):
                for j in range(i+2, len(res)):
                    diff = res[i] ^ res[j]
                    if bin(diff).count('1') == 1:
                        valid = False
                        break
                if not valid:
                    break
        print(f"Valid: {valid}")
    else:
        print("Valid: True")

