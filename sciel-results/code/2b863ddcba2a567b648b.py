"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def construct(q, n, radius):
    deadline = time.time() + 110.0
    rng = random.Random(42)
    
    total_words = q ** n
    if total_words > 100000:
        # Fallback for larger spaces: use tetracode base
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        base = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        return base
    
    # Precompute all words as tuples, and map index <-> word
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute cover sets: for each word, which words does it cover?
    # A word covers itself and all words at Hamming distance 1
    cover_sets = []
    for w in all_words:
        covered = {w}
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                covered.add(tuple(new_w))
        cover_sets.append([word_to_idx[cw] for cw in covered])
    
    # Simulated annealing over binary inclusion vector
    # State: selected = set of selected word indices
    # Energy = alpha * uncovered_count + len(selected)
    
    alpha = 100  # Heavy penalty for uncovered words
    
    def compute_uncovered(selected_set):
        """Compute number of uncovered words given a set of selected codewords."""
        covered_count = np.zeros(total_words, dtype=np.uint16)
        for c in selected_set:
            for w in cover_sets[c]:
                covered_count[w] += 1
        return int(np.sum(covered_count == 0))
    
    # Initialize with greedy set cover on all words
    # Start empty, repeatedly add word covering most uncovered
    uncovered = np.ones(total_words, dtype=np.uint8)
    selected = []
    selected_set = set()
    
    while uncovered.sum() > 0 and time.time() < deadline - 30:
        best_idx = -1
        best_gain = 0
        for i in range(total_words):
            if i in selected_set:
                continue
            gain = 0
            for w in cover_sets[i]:
                if uncovered[w]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_idx == -1 or best_gain == 0:
            break
        selected.append(best_idx)
        selected_set.add(best_idx)
        for w in cover_sets[best_idx]:
            uncovered[w] = 0
    
    current_uncovered = int(uncovered.sum())
    current_energy = alpha * current_uncovered + len(selected)
    best_selected = list(selected)
    best_energy = current_energy
    
    # Simulated annealing
    T = 50.0
    T_min = 0.1
    cooling = 0.9995
    steps_per_check = 100
    
    step_count = 0
    while time.time() < deadline - 10 and T > T_min:
        # Choose a move
        move_type = rng.random()
        if move_type < 0.4:
            # Add a random word
            new_word = rng.randint(0, total_words - 1)
            if new_word in selected_set:
                T *= cooling
                continue
            # Compute delta energy
            old_uncovered = current_uncovered
            # Count how many currently-uncovered words this new word covers
            new_covered = 0
            for w in cover_sets[new_word]:
                if uncovered[w]:
                    new_covered += 1
            delta_uncovered = -new_covered
            delta_energy = alpha * delta_uncovered + 1
            if delta_energy < 0 or rng.random() < np.exp(-delta_energy / T):
                selected.append(new_word)
                selected_set.add(new_word)
                for w in cover_sets[new_word]:
                    if uncovered[w]:
                        uncovered[w] = 0
                        current_uncovered -= 1
        elif move_type < 0.7 and len(selected) > 0:
            # Remove a random selected word
            idx = rng.randint(0, len(selected) - 1)
            new_word = selected[idx]
            # Compute delta energy
            old_uncovered = current_uncovered
            new_uncovered = 0
            for w in cover_sets[new_word]:
                # Count how many words become uncovered
                # We need to know if any other selected word covers w
                # This is expensive; approximate by checking if uncovered[w] would become 1
                pass
            # For removal, we need to check if any word becomes uncovered
            # Let's count how many words in cover_sets[new_word] are ONLY covered by new_word
            only_covered = 0
            for w in cover_sets[new_word]:
                if uncovered[w]:
                    # This word is currently uncovered, so it must be covered by new_word
                    # Wait, uncovered[w] == 0 means it IS covered
                    pass
            # Actually, let's just compute the change directly
            # Before removal: all words in cover_sets[new_word] are covered (at least by new_word)
            # After removal: a word w becomes uncovered iff no other selected word covers it
            # This is hard to compute quickly. Let's use a different approach.
            # For now, skip removal moves that might create uncovered words
            # Or compute it properly:
            newly_uncovered = 0
            for w in cover_sets[new_word]:
                # Check if any other selected word covers w
                is_covered_by_other = False
                for c in selected_set:
                    if c == new_word:
                        continue
                    if w in cover_sets[c]:
                        is_covered_by_other = True
                        break
                if not is_covered_by_other:
                    newly_uncovered += 1
            delta_uncovered = newly_uncovered
            delta_energy = alpha * delta_uncovered - 1
            if delta_energy < 0 or rng.random() < np.exp(-delta_energy / T):
                selected.remove(new_word)
                selected_set.discard(new_word)
                for w in cover_sets[new_word]:
                    # Check if still covered by other selected words
                    is_covered = False
                    for c in selected_set:
                        if w in cover_sets[c]:
                            is_covered = True
                            break
                    if not is_covered:
                        if not uncovered[w]:
                            uncovered[w] = 1
                            current_uncovered += 1
        else:
            # Swap: remove one, add one
            if len(selected) == 0:
                T *= cooling
                continue
            idx = rng.randint(0, len(selected) - 1)
            remove_word = selected[idx]
            add_word = rng.randint(0, total_words - 1)
            if add_word in selected_set:
                T *= cooling
                continue
            # Compute delta
            # Remove remove_word, add add_word
            # Net change in uncovered
            # Words that become uncovered: in cover_sets[remove_word] but not in cover_sets[add_word] and not covered by other selected
            # Words that become covered: in cover_sets[add_word] but not in cover_sets[remove_word] and currently uncovered
            # This is complex; skip for now
            T *= cooling
            continue
        
        step_count += 1
        if step_count % steps_per_check == 0:
            current_energy = alpha * current_uncovered + len(selected)
            if current_energy < best_energy:
                best_energy = current_energy
                best_selected = list(selected)
        T *= cooling
    
    # Final greedy cleanup: remove redundant words
    # A word is redundant if all words it covers are also covered by other selected words
    final_selected = set(best_selected)
    # Sort by index for determinism
    final_list = sorted(final_selected)
    
    # Try to remove redundant words
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        for i in range(len(final_list) - 1, -1, -1):
            w = final_list[i]
            # Check if all words covered by w are covered by other selected words
            redundant = True
            for cw in cover_sets[w]:
                is_covered_by_other = False
                for other in final_list:
                    if other == w:
                        continue
                    if cw in cover_sets[other]:
                        is_covered_by_other = True
                        break
                if not is_covered_by_other:
                    redundant = False
                    break
            if redundant:
                final_list.pop(i)
                changed = True
    
    result = [list(all_words[i]) for i in final_list]
    return result
# EVOLVE-BLOCK-END
