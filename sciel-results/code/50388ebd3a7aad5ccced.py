import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    def greedy():
        """Greedy: pick smallest valid x from 1..256."""
        values, sums = [], set()
        for x in range(1, 257):
            tails, total = [x], x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        return values

    best_seq = greedy()
    best_len = len(best_seq)

    # Fast penalty using prefix sums
    def penalty(seq):
        L = len(seq)
        prefix = [0] * (L + 1)
        for i in range(L):
            prefix[i + 1] = prefix[i] + seq[i]
        counts = {}
        for i in range(L):
            for j in range(i + 1, L + 1):
                s = prefix[j] - prefix[i]
                counts[s] = counts.get(s, 0) + 1
        return sum(c - 1 for c in counts.values() if c > 1)

    def random_seq(L):
        return sorted(random.sample(range(1, 257), L))

    # Simulated annealing on the sequence directly
    def sa(seq, time_limit):
        L = len(seq)
        cur = list(seq)
        cur_pen = penalty(cur)
        if cur_pen == 0:
            return cur, 0
        best = list(cur)
        best_pen = cur_pen
        T = 5.0
        T_min = 0.01
        cool = 0.9995
        t0 = time.time()
        while time.time() - t0 < time_limit:
            i = random.randint(0, L - 1)
            lo = cur[i - 1] + 1 if i > 0 else 1
            hi = cur[i + 1] - 1 if i < L - 1 else 256
            if lo > hi:
                continue
            v = random.randint(lo, hi)
            if v == cur[i]:
                continue
            old = cur[i]
            cur[i] = v
            np_ = penalty(cur)
            d = np_ - cur_pen
            if d < 0 or random.random() < math.exp(-d / T):
                cur_pen = np_
                if np_ < best_pen:
                    best_pen = np_
                    best = list(cur)
            else:
                cur[i] = old
            T = max(T * cool, T_min)
        return best, best_pen

    # Try to extend length
    target = best_len + 1
    while time.time() < deadline - 3.0:
        chunk = min(8.0, deadline - 3.0 - time.time())
        if chunk < 1.0:
            break
        # Mix of random and greedy-extended starts
        if random.random() < 0.5 and len(best_seq) >= target - 1:
            # Insert a random element into best_seq to get length target
            base = list(best_seq)
            # Pick a position to insert
            pos = random.randint(0, len(base))
            # Determine valid range for inserted element
            lo = base[pos - 1] + 1 if pos > 0 else 1
            hi = base[pos] - 1 if pos < len(base) else 256
            if lo <= hi:
                v = random.randint(lo, hi)
                start = base[:pos] + [v] + base[pos:]
            else:
                start = random_seq(target)
        else:
            start = random_seq(target)
        
        res, pen = sa(start, chunk)
        if pen == 0 and len(res) > best_len:
            best_seq = res
            best_len = len(best_seq)
            target = best_len + 1

    return best_seq
