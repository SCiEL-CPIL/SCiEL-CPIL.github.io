# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    deadline = time.time() + 110  # safety margin before 150s kill

    moduli = list(range(20, 201))
    nmod = len(moduli)
    idx_of = {m: i for i, m in enumerate(moduli)}

    # Precompute gcd table and per-modulus residue-class bitmasks
    gcd_table = [[0] * nmod for _ in range(nmod)]
    class_masks = [[] for _ in range(nmod)]
    for i, m in enumerate(moduli):
        masks = []
        for j, m2 in enumerate(moduli):
            g = math.gcd(m, m2)
            gcd_table[i][j] = g
            masks.append([1 << (r % g) for r in range(m)])
        class_masks[i] = masks

    def check_disjoint(i, r, classes):
        """classes: list of (idx, residue). Returns True if (moduli[i], r) is disjoint."""
        masks = class_masks[i]
        for j, r2 in classes:
            if (masks[j][r] & class_masks[j][r2]) != 0:
                return False
        return True

    def score_of(classes):
        return sum(1.0 / moduli[i] for i, r in classes)

    def greedy_randomized():
        classes = []
        for i, m in enumerate(moduli):
            residues = list(range(m))
            random.shuffle(residues)
            for r in residues:
                if check_disjoint(i, r, classes):
                    classes.append((i, r))
                    break
        return classes

    def local_search(classes):
        classes = [c for c in classes]
        current_score = score_of(classes)
        in_set = [False] * nmod
        for i, r in classes:
            in_set[i] = True

        improved = True
        while improved:
            improved = False
            # Pass A: add any missing modulus that fits
            for i, m in enumerate(moduli):
                if in_set[i]:
                    continue
                for r in range(m):
                    if check_disjoint(i, r, classes):
                        classes.append((i, r))
                        in_set[i] = True
                        current_score += 1.0 / m
                        improved = True
                        break
            # Pass B: 2-for-1 swap -- drop one class, try to add two others
            for idx in range(len(classes)):
                i_drop, r_drop = classes[idx]
                base = classes[:idx] + classes[idx + 1:]
                base_score = current_score - 1.0 / moduli[i_drop]
                base_in = in_set[:]
                base_in[i_drop] = False
                added = []
                for i, m in enumerate(moduli):
                    if base_in[i]:
                        continue
                    for r in range(m):
                        if check_disjoint(i, r, base + added):
                            added.append((i, r))
                            break
                if len(added) >= 2:
                    new_score = base_score + sum(1.0 / moduli[i] for i, r in added)
                    if new_score > current_score + 1e-12:
                        new_classes = base + added
                        current_score = new_score
                        in_set = [False] * nmod
                        for i, r in new_classes:
                            in_set[i] = True
                        classes = new_classes
                        improved = True
                        break
        return classes, current_score

    best_score = 0.0
    best_classes = []

    while time.time() < deadline:
        candidate = greedy_randomized()
        improved_candidate, score = local_search(candidate)
        if score > best_score:
            best_score = score
            best_classes = improved_candidate
        if best_score > 2.99:
            # near theoretical max; keep polishing with a few more restarts
            pass

    result = [[moduli[i], r] for i, r in best_classes]
    return result
# EVOLVE-BLOCK-END
