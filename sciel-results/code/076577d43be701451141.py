"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 110

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
    num_primes = len(primes)

    def get_factor_mask(n):
        mask = 0
        tmp = n
        for i, p in enumerate(primes):
            if p * p > tmp:
                break
            if tmp % p == 0:
                mask |= (1 << i)
                while tmp % p == 0:
                    tmp //= p
        if tmp > 1:
            mask |= (1 << prime_to_idx[tmp])
        return mask

    factor_masks = [0] * N
    for a in range(1, N):
        factor_masks[a] = get_factor_mask(a)

    scores = [0.0] * N
    for a in range(1, N):
        scores[a] = 1.0 / (N - a)

    def npf(a):
        return bin(factor_masks[a]).count('1')

    def compute_score(chosen):
        return sum(scores[a] for a in chosen)

    def greedy_construct(ordering):
        chosen = []
        chosen_set = set()
        chosen_mask = 0
        for a in ordering:
            if factor_masks[a] & chosen_mask == 0:
                chosen.append(a)
                chosen_set.add(a)
                chosen_mask |= factor_masks[a]
        return chosen, chosen_set, chosen_mask

    best_chosen = None
    best_score = -1.0
    best_chosen_set = None
    best_chosen_mask = 0

    def update_best(chosen, chosen_set, chosen_mask, score):
        nonlocal best_chosen, best_score, best_chosen_set, best_chosen_mask
        if score > best_score:
            best_score = score
            best_chosen = chosen
            best_chosen_set = chosen_set
            best_chosen_mask = chosen_mask

    # Strategy 1: Pure score descending
    ordering1 = list(range(511, 0, -1))
    c1, s1, m1 = greedy_construct(ordering1)
    sc1 = compute_score(c1)
    update_best(c1, s1, m1, sc1)

    # Strategy 2: Score / num_prime_factors
    ordering2 = sorted(range(1, 512), key=lambda x: scores[x] / max(1, npf(x)), reverse=True)
    c2, s2, m2 = greedy_construct(ordering2)
    sc2 = compute_score(c2)
    update_best(c2, s2, m2, sc2)

    # Strategy 3: More randomized greedy seeds
    for trial in range(50):
        if time.time() > deadline:
            break
        ordering = list(range(1, 512))
        random.shuffle(ordering)
        weighted = [(random.random() * 0.3 + scores[x] / max(1, npf(x)), x) for x in ordering]
        weighted.sort(reverse=True)
        ordering_w = [x for _, x in weighted]
        cw, sw, mw = greedy_construct(ordering_w)
        scw = compute_score(cw)
        update_best(cw, sw, mw, scw)

    # Best-improvement local search with multiple move types
    def best_improvement_ls(chosen, chosen_set, chosen_mask, current_score, max_rounds=30):
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            
            best_gain = 0.0
            best_move = None  # (move_type, args)
            
            # Try 1-add: add one element if possible
            for b in range(511, 0, -1):
                if b in chosen_set:
                    continue
                if factor_masks[b] & chosen_mask:
                    continue
                gain = scores[b]
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_move = ('add', (b,))
            
            # Try 1-swap: remove one, add one
            for a in chosen:
                temp_mask = chosen_mask & ~factor_masks[a]
                for b in range(511, 0, -1):
                    if b in chosen_set:
                        continue
                    if factor_masks[b] & temp_mask:
                        continue
                    gain = scores[b] - scores[a]
                    if gain > best_gain + 1e-12:
                        best_gain = gain
                        best_move = ('swap', (a, b))
            
            # Try 2-swap: remove two, add two
            n = len(chosen)
            if n >= 2:
                for i in range(n):
                    if time.time() > deadline:
                        break
                    a1 = chosen[i]
                    for j in range(i + 1, n):
                        a2 = chosen[j]
                        temp_mask = chosen_mask & ~factor_masks[a1] & ~factor_masks[a2]
                        valid = []
                        for b in range(511, 0, -1):
                            if b in chosen_set:
                                continue
                            if factor_masks[b] & temp_mask:
                                continue
                            valid.append(b)
                        for idx1 in range(len(valid)):
                            b1 = valid[idx1]
                            for idx2 in range(idx1 + 1, len(valid)):
                                b2 = valid[idx2]
                                if factor_masks[b1] & factor_masks[b2]:
                                    continue
                                gain = scores[b1] + scores[b2] - scores[a1] - scores[a2]
                                if gain > best_gain + 1e-12:
                                    best_gain = gain
                                    best_move = ('2swap', (a1, a2, b1, b2))
            
            if best_move is None:
                break
            
            mt = best_move[0]
            args = best_move[1]
            
            if mt == 'add':
                b = args[0]
                chosen.append(b)
                chosen_set.add(b)
                chosen_mask |= factor_masks[b]
                current_score += best_gain
            elif mt == 'swap':
                a, b = args
                chosen.remove(a)
                chosen.append(b)
                chosen_set.remove(a)
                chosen_set.add(b)
                chosen_mask = (chosen_mask & ~factor_masks[a]) | factor_masks[b]
                current_score += best_gain
            elif mt == '2swap':
                a1, a2, b1, b2 = args
                chosen.remove(a1)
                chosen.remove(a2)
                chosen.append(b1)
                chosen.append(b2)
                chosen_set.remove(a1)
                chosen_set.remove(a2)
                chosen_set.add(b1)
                chosen_set.add(b2)
                chosen_mask = (chosen_mask & ~factor_masks[a1] & ~factor_masks[a2]) | factor_masks[b1] | factor_masks[b2]
                current_score += best_gain
        
        return chosen, chosen_set, chosen_mask, current_score

    # Run best-improvement local search from the best greedy solution
    best_chosen, best_chosen_set, best_chosen_mask, best_score = best_improvement_ls(
        best_chosen, best_chosen_set, best_chosen_mask, best_score
    )

    # Simulated annealing for additional improvement
    def simulated_annealing(chosen, chosen_set, chosen_mask, current_score):
        temperature = 0.008
        cooling_rate = 0.999
        min_temperature = 0.000005

        while temperature > min_temperature and time.time() < deadline:
            move_type = random.random()
            if move_type < 0.5 and len(chosen) > 0:
                a = random.choice(chosen)
                temp_mask = chosen_mask & ~factor_masks[a]
                candidates = [b for b in range(1, 512) if b not in chosen_set and (factor_masks[b] & temp_mask) == 0]
                if not candidates:
                    temperature *= cooling_rate
                    continue
                b = random.choice(candidates)
                delta = scores[b] - scores[a]
                if delta > 0 or random.random() < math.exp(delta / temperature):
                    chosen.remove(a)
                    chosen.append(b)
                    chosen_set.remove(a)
                    chosen_set.add(b)
                    chosen_mask = (chosen_mask & ~factor_masks[a]) | factor_masks[b]
                    current_score += delta
            elif move_type < 0.8 and len(chosen) >= 2:
                a1, a2 = random.sample(chosen, 2)
                temp_mask = chosen_mask & ~factor_masks[a1] & ~factor_masks[a2]
                candidates = [b for b in range(1, 512) if b not in chosen_set and (factor_masks[b] & temp_mask) == 0]
                if len(candidates) < 2:
                    temperature *= cooling_rate
                    continue
                b1, b2 = random.sample(candidates, 2)
                if (factor_masks[b1] & factor_masks[b2]) != 0:
                    temperature *= cooling_rate
                    continue
                delta = scores[b1] + scores[b2] - scores[a1] - scores[a2]
                if delta > 0 or random.random() < math.exp(delta / temperature):
                    chosen.remove(a1)
                    chosen.remove(a2)
                    chosen.append(b1)
                    chosen.append(b2)
                    chosen_set.remove(a1)
                    chosen_set.remove(a2)
                    chosen_set.add(b1)
                    chosen_set.add(b2)
                    chosen_mask = (chosen_mask & ~factor_masks[a1] & ~factor_masks[a2]) | factor_masks[b1] | factor_masks[b2]
                    current_score += delta
            elif move_type < 0.9:
                # Try adding an element
                candidates = [b for b in range(1, 512) if b not in chosen_set and (factor_masks[b] & chosen_mask) == 0]
                if not candidates:
                    temperature *= cooling_rate
                    continue
                b = random.choice(candidates)
                delta = scores[b]
                if delta > 0 or random.random() < math.exp(delta / temperature):
                    chosen.append(b)
                    chosen_set.add(b)
                    chosen_mask |= factor_masks[b]
                    current_score += delta

            temperature *= cooling_rate

        return chosen, chosen_set, chosen_mask, current_score

    best_chosen, best_chosen_set, best_chosen_mask, best_score = simulated_annealing(
        best_chosen, best_chosen_set, best_chosen_mask, best_score
    )

    # Final best-improvement local search pass
    best_chosen, best_chosen_set, best_chosen_mask, best_score = best_improvement_ls(
        best_chosen, best_chosen_set, best_chosen_mask, best_score
    )

    return best_chosen
# EVOLVE-BLOCK-END
