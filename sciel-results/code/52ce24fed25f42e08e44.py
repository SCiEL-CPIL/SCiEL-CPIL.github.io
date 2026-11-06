import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 90s kill

    def is_time_up():
        return time.time() > deadline

    def structural_construct():
        """Super-increasing gaps guarantee distinct consecutive block sums."""
        best = []
        for a1 in range(1, 50):
            seq = [a1]
            a = a1
            sum_gaps = 0
            while True:
                min_gap = sum_gaps + 1 if sum_gaps > 0 else 1
                if a + min_gap > 256:
                    break
                a += min_gap
                sum_gaps += min_gap
                seq.append(a)
            if len(seq) > len(best):
                best = seq[:]
        return best

    def greedy_construct(seed=None, strategy="smallest"):
        """
        Fast greedy: pick next element > last such that all new consecutive
        block sums are distinct from existing ones.
        strategy: 'smallest', 'random', 'biased_small', 'biased_large'
        """
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()

        seq = []
        prefix_sums = [0]
        forbidden_sums = set()
        last = 0

        for _ in range(256):
            valid_candidates = []
            for x in range(last + 1, 257):
                new_prefix = prefix_sums[-1] + x
                ok = True
                for p in prefix_sums:
                    d = new_prefix - p
                    if d in forbidden_sums:
                        ok = False
                        break
                if ok:
                    valid_candidates.append(x)

            if not valid_candidates:
                break

            # Selection strategy
            if strategy == "smallest":
                x = valid_candidates[0]
            elif strategy == "random":
                x = rng.choice(valid_candidates)
            elif strategy == "biased_small":
                # 70% smallest, 20% from first quarter, 10% random
                r = rng.random()
                n = len(valid_candidates)
                if r < 0.7:
                    x = valid_candidates[0]
                elif r < 0.9:
                    x = valid_candidates[rng.randint(0, max(0, n // 4 - 1))]
                else:
                    x = rng.choice(valid_candidates)
            elif strategy == "biased_large":
                # 30% smallest, 40% from second half, 30% random
                r = rng.random()
                n = len(valid_candidates)
                if r < 0.3:
                    x = valid_candidates[0]
                elif r < 0.7:
                    x = valid_candidates[rng.randint(n // 2, n - 1)]
                else:
                    x = rng.choice(valid_candidates)
            else:
                x = valid_candidates[0]

            seq.append(x)
            last = x
            new_prefix = prefix_sums[-1] + x
            prefix_sums.append(new_prefix)
            for p in prefix_sums[:-1]:
                forbidden_sums.add(new_prefix - p)

        return seq

    def extend_by_replacement(seq, seed=None):
        """
        Try to extend a sequence by replacing the last k elements with
        different choices, then greedily continuing.
        """
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()

        best = seq[:]
        best_len = len(seq)

        for k in range(1, min(5, len(seq)) + 1):
            base = seq[:-k]
            if not base:
                continue

            # Rebuild state from base
            prefix_sums = [0]
            forbidden_sums = set()
            last = 0
            for a in base:
                new_prefix = prefix_sums[-1] + a
                for p in prefix_sums:
                    forbidden_sums.add(new_prefix - p)
                prefix_sums.append(new_prefix)
                last = a

            # Now greedily continue from base, but try multiple random paths
            for attempt in range(3):
                trial_seq = base[:]
                trial_prefix = prefix_sums[:]
                trial_forbidden = forbidden_sums.copy()
                trial_last = last

                for _ in range(256):
                    valid_candidates = []
                    for x in range(trial_last + 1, 257):
                        new_prefix = trial_prefix[-1] + x
                        ok = True
                        for p in trial_prefix:
                            d = new_prefix - p
                            if d in trial_forbidden:
                                ok = False
                                break
                        if ok:
                            valid_candidates.append(x)

                    if not valid_candidates:
                        break

                    # Randomized choice
                    if attempt == 0:
                        x = valid_candidates[0]
                    else:
                        x = rng.choice(valid_candidates)

                    trial_seq.append(x)
                    trial_last = x
                    new_prefix = trial_prefix[-1] + x
                    trial_prefix.append(new_prefix)
                    for p in trial_prefix[:-1]:
                        trial_forbidden.add(new_prefix - p)

                if len(trial_seq) > best_len:
                    best_len = len(trial_seq)
                    best = trial_seq[:]

        return best

    # Phase 1: Fast structural baseline
    best_seq = structural_construct()
    best_len = len(best_seq)

    # Phase 2: Many randomized greedy attempts
    strategies = ["smallest", "biased_small", "random", "biased_large"]
    attempt = 0
    while not is_time_up():
        strategy = strategies[attempt % len(strategies)]
        seq = greedy_construct(seed=attempt, strategy=strategy)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]

        attempt += 1

    # Phase 3: Try to extend the best sequence by modifying its tail
    if not is_time_up():
        for seed in range(200):
            if is_time_up():
                break
            extended = extend_by_replacement(best_seq, seed=seed)
            if len(extended) > best_len:
                best_len = len(extended)
                best_seq = extended[:]

    # Phase 4: If we still have time, run more greedy with fresh seeds
    if not is_time_up():
        attempt = 1000
        while not is_time_up():
            strategy = strategies[attempt % len(strategies)]
            seq = greedy_construct(seed=attempt, strategy=strategy)
            if len(seq) > best_len:
                best_len = len(seq)
                best_seq = seq[:]
            attempt += 1

    return best_seq
