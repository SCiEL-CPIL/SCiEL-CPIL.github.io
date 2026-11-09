"""Evolutionary strategy with delta-scoring for 3-term AP difference maximization."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    N = 32
    MAX_VAL = 512
    MAX_D = 255  # max d such that a + 2*d <= 512 for some a >= 1

    # ---------- Fast scoring utilities ----------

    def score_mask(mask):
        """Count distinct d > 0 with a 3-term AP in mask. O(255) bitwise ops."""
        count = 0
        for d in range(1, MAX_D + 1):
            if (mask & (mask >> d) & (mask >> 2 * d)) != 0:
                count += 1
        return count

    def diff_status(mask, d):
        """True if d has a 3-term AP in mask."""
        return (mask & (mask >> d) & (mask >> 2 * d)) != 0

    def affected_diffs(old_mask, new_mask):
        """
        Return set of d values whose AP-status may have changed.
        A value v affects diff d if v is part of some triple (v, v+d, v+2d),
        (v-d, v, v+d), or (v-2d, v-d, v).
        So d must satisfy: d divides the gap to another set element in a specific way.
        Simpler: d is affected if any of v, v±d, v±2d was in old or new mask.
        We return all d in 1..MAX_D where the triple status could change.
        """
        changed = old_mask ^ new_mask
        if changed == 0:
            return set()
        # For each changed bit position v, d is affected if:
        #   (v in mask) and (v+d in mask) and (v+2d in mask)  -> v is 'a'
        #   (v-d in mask) and (v in mask) and (v+d in mask)  -> v is 'a+d'
        #   (v-2d in mask) and (v-d in mask) and (v in mask) -> v is 'a+2d'
        # So d is affected if any of v, v±d, v±2d is in (old_mask | new_mask)
        # AND d is such that the triple is within 1..512.
        # Efficient: collect candidate d's from changed positions.
        affected = set()
        for v in range(1, MAX_VAL + 1):
            if (changed >> v) & 1:
                # v changed; d is affected if v±d or v±2d is in the union mask
                union = old_mask | new_mask
                for d in range(1, MAX_D + 1):
                    # Check if any triple member involving v and step d is in union
                    # v as a: need v+d, v+2d in union
                    # v as a+d: need v-d, v+d in union
                    # v as a+2d: need v-2d, v-d in union
                    if d < MAX_D:
                        if (union >> (v + d)) & 1 if v + d <= MAX_VAL else False:
                            affected.add(d)
                        if (union >> (v + 2 * d)) & 1 if v + 2 * d <= MAX_VAL else False:
                            affected.add(d)
                    if d <= v - 1:
                        if (union >> (v - d)) & 1:
                            affected.add(d)
                    if 2 * d <= v - 1:
                        if (union >> (v - 2 * d)) & 1:
                            affected.add(d)
        return affected

    def delta_score(old_mask, new_mask, old_score):
        """
        Compute new score from old by only rechecking affected diffs.
        """
        affected = affected_diffs(old_mask, new_mask)
        if not affected:
            return old_score
        score = old_score
        for d in affected:
            was = diff_status(old_mask, d)
            now = diff_status(new_mask, d)
            if was and not now:
                score -= 1
            elif now and not was:
                score += 1
        return score

    # ---------- Population representation ----------

    def mask_to_list(mask):
        return [i for i in range(1, MAX_VAL + 1) if (mask >> i) & 1]

    def list_to_mask(lst):
        m = 0
        for v in lst:
            m |= (1 << v)
        return m

    def random_mask():
        lst = random.sample(range(1, MAX_VAL + 1), N)
        return list_to_mask(lst)

    def mutate_mask(mask):
        """
        Generate a mutated offspring:
        - With prob 0.5: replace 1 random element
        - With prob 0.3: replace 2 random elements
        - With prob 0.2: swap 2 elements
        """
        lst = mask_to_list(mask)
        r = random.random()
        if r < 0.5:
            # Replace 1
            i = random.randint(0, N - 1)
            old_val = lst[i]
            new_val = random.randint(1, MAX_VAL)
            if new_val == old_val:
                new_val = (new_val % MAX_VAL) + 1
            lst[i] = new_val
            # Ensure uniqueness
            if lst.count(new_val) > 1:
                # Pick another
                while lst.count(new_val) > 1:
                    new_val = random.randint(1, MAX_VAL)
                lst[i] = new_val
        elif r < 0.8:
            # Replace 2
            i, j = random.sample(range(N), 2)
            lst[i] = random.randint(1, MAX_VAL)
            lst[j] = random.randint(1, MAX_VAL)
            # Fix duplicates
            seen = set()
            for k in range(N):
                while lst[k] in seen or not (1 <= lst[k] <= MAX_VAL):
                    lst[k] = random.randint(1, MAX_VAL)
                seen.add(lst[k])
        else:
            # Swap 2
            i, j = random.sample(range(N), 2)
            lst[i], lst[j] = lst[j], lst[i]
        return list_to_mask(lst)

    def crossover_mask(m1, m2):
        """Uniform crossover on element lists."""
        l1 = mask_to_list(m1)
        l2 = mask_to_list(m2)
        child = []
        for i in range(N):
            if random.random() < 0.5:
                child.append(l1[i])
            else:
                child.append(l2[i])
        # Fix duplicates
        seen = set()
        for i in range(N):
            while child[i] in seen or not (1 <= child[i] <= MAX_VAL):
                child[i] = random.randint(1, MAX_VAL)
            seen.add(child[i])
        return list_to_mask(child)

    # ---------- Evolutionary Strategy ----------

    POP_SIZE = 200
    ELITE = 10
    TOURNAMENT_K = 5

    # Initialize population
    pop = []
    scores = []
    for _ in range(POP_SIZE):
        m = random_mask()
        s = score_mask(m)
        pop.append(m)
        scores.append(s)

    # Sort by score descending
    order = sorted(range(POP_SIZE), key=lambda i: scores[i], reverse=True)
    pop = [pop[i] for i in order]
    scores = [scores[i] for i in order]

    best_mask = pop[0]
    best_score = scores[0]

    # Also try structured initializations
    structured_starts = []
    # Arithmetic progression
    for base in range(1, 20):
        for step in range(2, 16):
            lst = []
            for i in range(N):
                v = base + i * step
                if v <= MAX_VAL:
                    lst.append(v)
            if len(lst) == N:
                structured_starts.append(lst)
    # Two interleaved APs
    for b1 in range(1, 30):
        for b2 in range(30, 60):
            for s1 in range(6, 14):
                for s2 in range(6, 14):
                    lst = []
                    for i in range(N // 2):
                        v1 = b1 + i * s1
                        v2 = b2 + i * s2
                        if v1 <= MAX_VAL:
                            lst.append(v1)
                        if v2 <= MAX_VAL:
                            lst.append(v2)
                    if len(lst) >= N:
                        lst = lst[:N]
                        structured_starts.append(lst)

    # Add best structured starts to population
    for lst in structured_starts[:50]:
        m = list_to_mask(lst)
        s = score_mask(m)
        if s > scores[-1]:
            pop[-1] = m
            scores[-1] = s
            order = sorted(range(POP_SIZE), key=lambda i: scores[i], reverse=True)
            pop = [pop[i] for i in order]
            scores = [scores[i] for i in order]
            if s > best_score:
                best_score = s
                best_mask = m

    # Evolutionary loop
    generation = 0
    while time.time() < deadline - 5.0:
        generation += 1
        new_pop = []
        new_scores = []

        # Elitism: keep top ELITE
        for i in range(ELITE):
            new_pop.append(pop[i])
            new_scores.append(scores[i])

        # Generate offspring to fill population
        while len(new_pop) < POP_SIZE:
            if time.time() > deadline - 5.0:
                break

            # Tournament selection for parents
            idx1 = random.randint(0, min(ELITE * 3, POP_SIZE) - 1)
            idx2 = random.randint(0, min(ELITE * 3, POP_SIZE) - 1)
            p1 = pop[idx1]
            p2 = pop[idx2]

            r = random.random()
            if r < 0.4:
                child = crossover_mask(p1, p2)
            elif r < 0.8:
                child = mutate_mask(p1)
            else:
                child = mutate_mask(mutate_mask(p2))  # double mutation for diversity

            # Delta-score against parent
            parent_score = scores[idx1] if random.random() < 0.5 else scores[idx2]
            parent_mask = p1 if random.random() < 0.5 else p2
            child_score = delta_score(parent_mask, child, parent_score)

            # If delta-score is unreliable (large change), verify
            if random.random() < 0.05:  # 5% verification
                child_score = score_mask(child)

            new_pop.append(child)
            new_scores.append(child_score)

            if child_score > best_score:
                best_score = child_score
                best_mask = child

        # Sort new population
        order = sorted(range(len(new_pop)), key=lambda i: new_scores[i], reverse=True)
        pop = [new_pop[i] for i in order]
        scores = [new_scores[i] for i in order]

        # Periodic full re-score to correct drift
        if generation % 50 == 0:
            for i in range(len(pop)):
                scores[i] = score_mask(pop[i])
            order = sorted(range(len(pop)), key=lambda i: scores[i], reverse=True)
            pop = [pop[i] for i in order]
            scores = [scores[i] for i in order]
            best_score = scores[0]
            best_mask = pop[0]

    # Final verification
    final_score = score_mask(best_mask)
    if final_score > best_score:
        best_score = final_score

    result = mask_to_list(best_mask)
    # Ensure exactly 32 distinct in 1..512
    seen = set()
    clean = []
    for v in result:
        if 1 <= v <= MAX_VAL and v not in seen:
            seen.add(v)
            clean.append(v)
    while len(clean) < N:
        v = random.randint(1, MAX_VAL)
        if v not in seen:
            seen.add(v)
            clean.append(v)
    clean = clean[:N]
    return sorted(clean)

# EVOLVE-BLOCK-END
