import random
import time
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = 130.0
    n = 16

    # Precompute all 560 triples
    triple_list = []
    mask_to_idx = {}
    idx_to_pts = {}
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                mask = (1 << i) | (1 << j) | (1 << k)
                idx = len(triple_list)
                triple_list.append(mask)
                mask_to_idx[mask] = idx
                idx_to_pts[idx] = (i, j, k)
    num_triples = len(triple_list)

    # Conflicts: share >= 2 points
    conflicts = [set() for _ in range(num_triples)]
    for i in range(num_triples):
        mi = triple_list[i]
        for j in range(i + 1, num_triples):
            mj = triple_list[j]
            common = mi & mj
            if common & (common - 1):
                conflicts[i].add(j)
                conflicts[j].add(i)

    # All 5-sets and 4-sets with their contained triples
    masks_size_5 = []
    mask_5_to_triples = {}
    masks_size_4 = []
    mask_4_to_triples = {}
    for mask in range(1 << n):
        pop = bin(mask).count('1')
        if pop == 5:
            masks_size_5.append(mask)
            mask_5_to_triples[mask] = [t for t in range(num_triples) if (triple_list[t] & mask) == triple_list[t]]
        elif pop == 4:
            masks_size_4.append(mask)
            mask_4_to_triples[mask] = [t for t in range(num_triples) if (triple_list[t] & mask) == triple_list[t]]

    num_5 = len(masks_size_5)
    num_4 = len(masks_size_4)

    def calc_alpha_full(active):
        """Exact alpha via full scan of 5-sets then 4-sets."""
        for m in masks_size_5:
            covered = False
            for t in mask_5_to_triples[m]:
                if t in active:
                    covered = True
                    break
            if not covered:
                return 5
        for m in masks_size_4:
            covered = False
            for t in mask_4_to_triples[m]:
                if t in active:
                    covered = True
                    break
            if not covered:
                return 4
        if len(active) < num_triples:
            return 3
        return 2

    def greedy_pack():
        """Dense greedy: repeatedly add a triple with minimum conflicts."""
        active = set()
        remaining = list(range(num_triples))
        while remaining:
            best = -1
            best_conf = float('inf')
            for idx in remaining:
                c = 0
                for cf in conflicts[idx]:
                    if cf in active:
                        c += 1
                if c < best_conf:
                    best_conf = c
                    best = idx
                    if c == 0:
                        break
            if best < 0:
                break
            active.add(best)
            remaining.remove(best)
        return active

    def find_witness(active, target_size):
        """Find an independent set of target_size (5 or 4)."""
        if target_size == 5:
            for m in masks_size_5:
                covered = False
                for t in mask_5_to_triples[m]:
                    if t in active:
                        covered = True
                        break
                if not covered:
                    return m
        elif target_size == 4:
            for m in masks_size_4:
                covered = False
                for t in mask_4_to_triples[m]:
                    if t in active:
                        covered = True
                        break
                if not covered:
                    return m
        return None

    def get_local_triples(witness_mask):
        """All triples contained in witness_mask."""
        pts = [i for i in range(n) if witness_mask & (1 << i)]
        result = []
        for a in range(len(pts)):
            for b in range(a + 1, len(pts)):
                for c in range(b + 1, len(pts)):
                    m = (1 << pts[a]) | (1 << pts[b]) | (1 << pts[c])
                    idx = mask_to_idx.get(m)
                    if idx is not None:
                        result.append(idx)
        return result

    def targeted_2exchange(active, witness):
        """Try 1-add or 2-exchange to add a triple inside witness."""
        local = get_local_triples(witness)
        random.shuffle(local)
        for t in local:
            if t in active:
                continue
            # Try direct add
            can_add = True
            for cf in conflicts[t]:
                if cf in active:
                    can_add = False
                    break
            if can_add:
                new_active = set(active)
                new_active.add(t)
                return new_active
            else:
                # Try 2-exchange: remove one conflicting edge, add t
                confs_in_active = [cf for cf in conflicts[t] if cf in active]
                random.shuffle(confs_in_active)
                for cf in confs_in_active:
                    new_active = set(active)
                    new_active.remove(cf)
                    ok = True
                    for cf2 in conflicts[t]:
                        if cf2 in new_active:
                            ok = False
                            break
                    if ok:
                        new_active.add(t)
                        return new_active
        return None

    def random_2exchange(active):
        """Random 2-exchange: remove one, add one valid."""
        if not active:
            return None
        to_remove = random.choice(list(active))
        new_active = set(active)
        new_active.remove(to_remove)
        candidates = list(range(num_triples))
        random.shuffle(candidates)
        for idx in candidates:
            if idx in new_active:
                continue
            ok = True
            for cf in conflicts[idx]:
                if cf in new_active:
                    ok = False
                    break
            if ok:
                new_active.add(idx)
                return new_active
        return None

    best_alpha = n
    best_triples = None

    # Multiple greedy restarts
    for restart in range(8):
        if time.time() > deadline:
            break
        current = greedy_pack()
        alpha = calc_alpha_full(current)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = set(current)
        if best_alpha <= 4:
            break

    if best_triples is None:
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]

    current = set(best_triples)
    current_alpha = best_alpha

    # Local search with targeted moves
    while time.time() < deadline:
        if current_alpha < best_alpha:
            best_alpha = current_alpha
            best_triples = set(current)

        move_type = random.random()
        new_active = None

        if current_alpha >= 5 and move_type < 0.6:
            witness = find_witness(current, 5)
            if witness is not None:
                new_active = targeted_2exchange(current, witness)
        elif current_alpha == 4 and move_type < 0.6:
            witness = find_witness(current, 4)
            if witness is not None:
                new_active = targeted_2exchange(current, witness)

        if new_active is None:
            new_active = random_2exchange(current)

        if new_active is None:
            # Perturb: remove 1-2 random edges
            for _ in range(random.randint(1, 2)):
                if current:
                    current.remove(random.choice(list(current)))
            continue

        new_alpha = calc_alpha_full(new_active)
        delta = new_alpha - current_alpha
        if delta <= 0:
            current = new_active
            current_alpha = new_alpha
        elif delta == 1 and random.random() < 0.3:
            current = new_active
            current_alpha = new_alpha

    result = []
    for idx in best_triples:
        pts = list(idx_to_pts[idx])
        result.append(pts)
    return result
