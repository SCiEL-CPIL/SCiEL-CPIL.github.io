"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140

    N = 512

    def get_primes(n):
        sieve = [True] * n
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]

    primes = get_primes(N)
    prime_to_idx = {p: i for i, p in enumerate(primes)}

    masks = [0] * N
    scores = [0.0] * N
    for a in range(1, N):
        n = a
        mask = 0
        for i, p in enumerate(primes):
            if p * p > n:
                break
            if n % p == 0:
                mask |= (1 << i)
                while n % p == 0:
                    n //= p
        if n > 1:
            if n in prime_to_idx:
                mask |= (1 << prime_to_idx[n])
        masks[a] = mask
        scores[a] = 1.0 / (N - a)

    def greedy_seed(perturb=None):
        if perturb is None:
            w = scores
        else:
            w = [scores[a] * perturb.get(a, 1.0) for a in range(N)]
        candidates = sorted(range(1, N), key=lambda a: w[a], reverse=True)
        chosen = []
        chosen_set = set()
        used_mask = 0
        for a in candidates:
            if masks[a] & used_mask == 0:
                chosen.append(a)
                chosen_set.add(a)
                used_mask |= masks[a]
        return chosen, chosen_set, used_mask

    def local_search(chosen, chosen_set, used_mask, dl):
        cur_score = sum(scores[a] for a in chosen)
        non_chosen = [a for a in range(1, N) if a not in chosen_set]
        improved = True
        while improved and time.time() < dl:
            improved = False
            # ADD
            for idx, a in enumerate(non_chosen):
                if masks[a] & used_mask == 0:
                    chosen.append(a)
                    chosen_set.add(a)
                    used_mask |= masks[a]
                    cur_score += scores[a]
                    non_chosen.pop(idx)
                    improved = True
                    break
            if improved:
                continue
            # 1-1
            nc = len(chosen)
            for i in range(nc):
                if time.time() > dl:
                    break
                a = chosen[i]
                mw = used_mask ^ masks[a]
                best_b = -1
                best_g = 0.0
                for b in non_chosen:
                    if masks[b] & mw == 0:
                        g = scores[b] - scores[a]
                        if g > best_g:
                            best_g = g
                            best_b = b
                if best_b != -1 and best_g > 1e-12:
                    chosen[i] = best_b
                    chosen_set.discard(a)
                    chosen_set.add(best_b)
                    used_mask = mw | masks[best_b]
                    cur_score += best_g
                    non_chosen.remove(best_b)
                    non_chosen.append(a)
                    improved = True
                    break
            if improved:
                continue
            # 1-2
            nc = len(chosen)
            for i in range(nc):
                if time.time() > dl:
                    break
                a = chosen[i]
                mw = used_mask ^ masks[a]
                ts = scores[a]
                best_pair = None
                best_g = 0.0
                tnc = sorted(non_chosen, key=lambda x: scores[x], reverse=True)[:60]
                for bi in range(len(tnc)):
                    b = tnc[bi]
                    if masks[b] & mw:
                        continue
                    for di in range(bi+1, len(tnc)):
                        d = tnc[di]
                        if masks[d] & mw or (masks[b] & masks[d]):
                            continue
                        g = scores[b] + scores[d] - ts
                        if g > best_g:
                            best_g = g
                            best_pair = (b, d)
                if best_pair and best_g > 1e-12:
                    b, d = best_pair
                    chosen[i] = b
                    chosen.append(d)
                    chosen_set.discard(a)
                    chosen_set.add(b)
                    chosen_set.add(d)
                    used_mask = mw | masks[b] | masks[d]
                    cur_score += best_g
                    non_chosen.remove(b)
                    non_chosen.remove(d)
                    non_chosen.append(a)
                    improved = True
                    break
            if improved:
                continue
            # 2-1
            nc = len(chosen)
            found = False
            for i in range(nc):
                if time.time() > dl:
                    break
                a = chosen[i]
                for j in range(i+1, nc):
                    c = chosen[j]
                    mw = used_mask ^ masks[a] ^ masks[c]
                    ts = scores[a] + scores[c]
                    best_b = -1
                    best_g = 0.0
                    for b in non_chosen:
                        if masks[b] & mw == 0:
                            g = scores[b] - ts
                            if g > best_g:
                                best_g = g
                                best_b = b
                    if best_b != -1 and best_g > 1e-12:
                        # Replace positions i and j with best_b (shrink by 1)
                        chosen[i] = best_b
                        chosen[j] = chosen[-1]
                        chosen.pop()
                        chosen_set.discard(a)
                        chosen_set.discard(c)
                        chosen_set.add(best_b)
                        used_mask = mw | masks[best_b]
                        cur_score += best_g
                        non_chosen.remove(best_b)
                        non_chosen.append(a)
                        non_chosen.append(c)
                        improved = True
                        found = True
                        break
                if found:
                    break
            if improved:
                continue
            # 2-2
            nc = len(chosen)
            found = False
            for i in range(nc):
                if time.time() > dl:
                    break
                a = chosen[i]
                for j in range(i+1, nc):
                    c = chosen[j]
                    mw = used_mask ^ masks[a] ^ masks[c]
                    ts = scores[a] + scores[c]
                    best_pair = None
                    best_g = 0.0
                    tnc = sorted(non_chosen, key=lambda x: scores[x], reverse=True)[:60]
                    for bi in range(len(tnc)):
                        b = tnc[bi]
                        if masks[b] & mw:
                            continue
                        for di in range(bi+1, len(tnc)):
                            d = tnc[di]
                            if masks[d] & mw or (masks[b] & masks[d]):
                                continue
                            g = scores[b] + scores[d] - ts
                            if g > best_g:
                                best_g = g
                                best_pair = (b, d)
                    if best_pair and best_g > 1e-12:
                        b, d = best_pair
                        chosen[i] = b
                        chosen[j] = d
                        chosen_set.discard(a)
                        chosen_set.discard(c)
                        chosen_set.add(b)
                        chosen_set.add(d)
                        used_mask = mw | masks[b] | masks[d]
                        cur_score += best_g
                        non_chosen.remove(b)
                        non_chosen.remove(d)
                        non_chosen.append(a)
                        non_chosen.append(c)
                        improved = True
                        found = True
                        break
                if found:
                    break
        return chosen, cur_score

    best_chosen = None
    best_score = 0.0

    # Pure greedy
    chosen, cs, um = greedy_seed()
    chosen, sc = local_search(chosen, cs, um, deadline)
    if sc > best_score:
        best_score = sc
        best_chosen = chosen[:]

    # Randomized seeds
    random.seed(123)
    for s in range(20):
        if time.time() > deadline - 3:
            break
        perturb = {a: random.uniform(0.8, 1.2) for a in range(1, N)}
        chosen, cs, um = greedy_seed(perturb)
        chosen, sc = local_search(chosen, cs, um, deadline)
        if sc > best_score:
            best_score = sc
            best_chosen = chosen[:]

    return best_chosen if best_chosen else [1]
# EVOLVE-BLOCK-END
