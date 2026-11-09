import time
import random
import math
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    N = 32
    MAX_VAL = 512
    random.seed(12345)

    def score_mask(mask):
        count = 0
        for d in range(1, 256):
            if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
                count += 1
        return count

    def list_to_mask(lst):
        m = 0
        for x in lst:
            m |= (1 << x)
        return m

    def mask_to_list(mask):
        return [i + 1 for i in range(MAX_VAL) if (mask >> i) & 1]

    # Generate structured candidates with more variety
    def gen_structured_candidates():
        cands = []
        # Interval
        lst = list(range(1, 33))
        m = list_to_mask(lst)
        s = score_mask(m)
        cands.append((lst, m, s, 'interval'))
        # AP with various starts and steps
        for start in range(1, 20):
            for step in range(3, 20):
                lst = [start + i * step for i in range(N)]
                if max(lst) > MAX_VAL:
                    continue
                m = list_to_mask(lst)
                s = score_mask(m)
                cands.append((lst, m, s, 'AP'))
        # Two interleaved APs
        for step1 in range(3, 15):
            for step2 in range(3, 15):
                if step1 == step2:
                    continue
                lst = []
                for i in range(16):
                    lst.append(1 + i * step1)
                    lst.append(2 + i * step2)
                if len(set(lst)) != N or max(lst) > MAX_VAL:
                    continue
                m = list_to_mask(lst)
                s = score_mask(m)
                cands.append((lst, m, s, 'two_AP'))
        # Quadratic sequences
        for a in range(1, 5):
            for b in range(1, 10):
                lst = [a * i * i + b * i + 1 for i in range(N)]
                if max(lst) > MAX_VAL or len(set(lst)) != N:
                    continue
                m = list_to_mask(lst)
                s = score_mask(m)
                cands.append((lst, m, s, 'quad'))
        return cands

    # Random candidates
    def gen_random_candidates(count):
        cands = []
        for _ in range(count):
            lst = random.sample(range(1, MAX_VAL + 1), N)
            m = list_to_mask(lst)
            s = score_mask(m)
            cands.append((lst, m, s, 'random'))
        return cands

    # Sumset candidates: A + k*B
    def gen_sumset_candidates():
        cands = []
        for sizeA in range(2, 12):
            for sizeB in range(2, 12):
                if sizeA * sizeB > 60:
                    continue
                for mult in [2, 3, 4, 5, 6, 7, 8]:
                    for trial in range(300):
                        A = set(random.sample(range(1, 50), sizeA))
                        maxB = (MAX_VAL - 1) // mult
                        B = set(random.sample(range(1, maxB + 1), sizeB))
                        S = set()
                        for a in A:
                            for b in B:
                                v = a + mult * b
                                if 1 <= v <= MAX_VAL:
                                    S.add(v)
                        if len(S) == N:
                            lst = sorted(S)
                            m = list_to_mask(lst)
                            s = score_mask(m)
                            cands.append((lst, m, s, 'sumset'))
                        elif len(S) > N:
                            # Greedy trim: pick elements that contribute most to triples
                            lst_sorted = sorted(S)
                            best = []
                            bm = 0
                            # Try adding elements in order of their "potential"
                            potentials = []
                            for x in lst_sorted:
                                # Count how many existing best elements form triples with x
                                pot = 0
                                for y in best:
                                    d = abs(x - y)
                                    if d > 0:
                                        if (x + d) in S or (x - d) in S:
                                            pot += 1
                                potentials.append((pot, x))
                            potentials.sort(reverse=True)
                            for pot, x in potentials:
                                if len(best) >= N:
                                    break
                                best.append(x)
                                bm |= (1 << x)
                            if len(best) == N:
                                m = list_to_mask(best)
                                s = score_mask(m)
                                cands.append((best, m, s, 'sumset_trim'))
        return cands

    # Greedy construction
    def greedy_construct():
        current_mask = 0
        current_set = set()
        for _ in range(N):
            best_val = -1
            best_score = -1
            for x in range(1, MAX_VAL + 1):
                if x in current_set:
                    continue
                new_mask = current_mask | (1 << x)
                s = score_mask(new_mask)
                if s > best_score:
                    best_score = s
                    best_val = x
            if best_val != -1:
                current_set.add(best_val)
                current_mask |= (1 << best_val)
        return list(current_set), current_mask, best_score

    # Generate all candidates
    all_cands = []
    t0 = time.time()
    all_cands.extend(gen_structured_candidates())
    all_cands.extend(gen_random_candidates(100))
    all_cands.extend(gen_sumset_candidates())
    t1 = time.time()
    print(f"Generated {len(all_cands)} candidates in {t1-t0:.1f}s", file=sys.stderr)

    # Find best initial
    best_score = -1
    best_list = None
    best_mask = 0
    for lst, m, s, tag in all_cands:
        if s > best_score:
            best_score = s
            best_list = lst[:]
            best_mask = m
    print(f"Best initial score: {best_score}", file=sys.stderr)

    # Also add greedy
    g_list, g_mask, g_score = greedy_construct()
    if g_score > best_score:
        best_score = g_score
        best_list = g_list[:]
        best_mask = g_mask

    # Simulated Annealing with improved move generation
    current_list = best_list[:]
    current_score = best_score
    current_mask = best_mask
    current_set = set(current_list)

    T = 10.0
    cooling_rate = 0.9997
    no_improve_count = 0
    no_improve_threshold = 5000
    restart_temperature = 5.0
    step = 0

    while time.time() < deadline:
        step += 1
        if step % 20000 == 0:
            print(f"Step {step}, current={current_score}, best={best_score}, T={T:.3f}", file=sys.stderr)
        
        move_type = random.random()
        if move_type < 0.5:
            # Single change
            idx = random.randint(0, N - 1)
            old_val = current_list[idx]
            new_val = random.randint(1, MAX_VAL)
            if new_val == old_val or new_val in current_set:
                continue
            new_mask = current_mask ^ (1 << old_val) ^ (1 << new_val)
            new_score = score_mask(new_mask)
            delta = new_score - current_score
            if delta > 0 or (T > 1e-10 and random.random() < math.exp(delta / T)):
                current_list[idx] = new_val
                current_set.remove(old_val)
                current_set.add(new_val)
                current_score = new_score
                current_mask = new_mask
                if current_score > best_score:
                    best_score = current_score
                    best_list = current_list[:]
                    best_mask = current_mask
                    no_improve_count = 0
                else:
                    no_improve_count += 1
        elif move_type < 0.8:
            # Remove 1, Add 1 (different position)
            i = random.randint(0, N - 1)
            old_val = current_list[i]
            new_val = random.randint(1, MAX_VAL)
            if new_val == old_val or new_val in current_set:
                continue
            new_mask = current_mask ^ (1 << old_val) ^ (1 << new_val)
            new_score = score_mask(new_mask)
            delta = new_score - current_score
            if delta > 0 or (T > 1e-10 and random.random() < math.exp(delta / T)):
                current_list[i] = new_val
                current_set.remove(old_val)
                current_set.add(new_val)
                current_score = new_score
                current_mask = new_mask
                if current_score > best_score:
                    best_score = current_score
                    best_list = current_list[:]
                    best_mask = current_mask
                    no_improve_count = 0
                else:
                    no_improve_count += 1
        else:
            # Remove 2, Add 2
            i, j = random.sample(range(N), 2)
            old_val1 = current_list[i]
            old_val2 = current_list[j]
            new_val1 = -1
            new_val2 = -1
            attempts = 0
            while (new_val1 == -1 or new_val2 == -1) and attempts < 50:
                v1 = random.randint(1, MAX_VAL)
                v2 = random.randint(1, MAX_VAL)
                if new_val1 == -1 and v1 not in current_set and v1 != old_val1 and v1 != old_val2:
                    new_val1 = v1
                if new_val2 == -1 and v2 not in current_set and v2 != old_val1 and v2 != old_val2 and v2 != new_val1:
                    new_val2 = v2
                attempts += 1
            if new_val1 != -1 and new_val2 != -1:
                new_mask = current_mask ^ (1 << old_val1) ^ (1 << old_val2) ^ (1 << new_val1) ^ (1 << new_val2)
                new_score = score_mask(new_mask)
                delta = new_score - current_score
                if delta > 0 or (T > 1e-10 and random.random() < math.exp(delta / T)):
                    current_list[i] = new_val1
                    current_list[j] = new_val2
                    current_set.remove(old_val1)
                    current_set.remove(old_val2)
                    current_set.add(new_val1)
                    current_set.add(new_val2)
                    current_score = new_score
                    current_mask = new_mask
                    if current_score > best_score:
                        best_score = current_score
                        best_list = current_list[:]
                        best_mask = current_mask
                        no_improve_count = 0
                    else:
                        no_improve_count += 1

        T *= cooling_rate
        if T < 0.05:
            T = 0.05
        if no_improve_count > no_improve_threshold:
            current_list = best_list[:]
            current_score = best_score
            current_mask = best_mask
            current_set = set(current_list)
            T = restart_temperature
            no_improve_count = 0
            restart_temperature = max(1.0, restart_temperature * 0.7)

    # Final local search: try all single swaps for best improvement
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N):
            old_val = current_list[i]
            for new_val in range(1, MAX_VAL + 1):
                if new_val == old_val or new_val in current_set:
                    continue
                new_mask = current_mask ^ (1 << old_val) ^ (1 << new_val)
                new_score = score_mask(new_mask)
                if new_score > current_score:
                    current_list[i] = new_val
                    current_set.remove(old_val)
                    current_set.add(new_val)
                    current_score = new_score
                    current_mask = new_mask
                    if current_score > best_score:
                        best_score = current_score
                        best_list = current_list[:]
                        best_mask = current_mask
                    improved = True
                    break
    print(f"Final score: {best_score}", file=sys.stderr)
    return best_list
# EVOLVE-BLOCK-END
