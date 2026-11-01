import time
import random
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    # 1. Generate the naive superpermutation using standard construction
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)

        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = 0
            max_ov = min(len(new), len(block) - 1)
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new

    # 2. Initialize permutation counts
    perm_count = defaultdict(int)
    for i in range(len(word) - n + 1):
        p = tuple(word[i:i + n])
        perm_count[p] += 1

    # 3. Helper: check if position i is deletable
    def can_delete(i):
        L = len(word)
        if L <= n:
            return False
        start_j = max(0, i - n + 1)
        end_j = min(i, L - n)
        for j in range(start_j, end_j + 1):
            if perm_count[tuple(word[j:j + n])] == 1:
                return False
        return True

    # 4. Helper: delete position i and update counts
    def delete(i):
        L = len(word)
        start_j = max(0, i - n + 1)
        end_j = min(i, L - n)

        # Decrement old windows
        for j in range(start_j, end_j + 1):
            p = tuple(word[j:j + n])
            perm_count[p] -= 1

        # Commit deletion
        word.pop(i)

        # Increment new windows
        new_L = L - 1
        start_j_new = max(0, i - n + 1)
        end_j_new = min(i, new_L - n)

        for j in range(start_j_new, end_j_new + 1):
            new_w = tuple(word[j:j + n])
            perm_count[new_w] += 1

    # 5. Helper: check if swap (i, j) is valid
    def can_swap(i, j):
        if i == j or word[i] == word[j]:
            return False
        L = len(word)
        if L <= n:
            return False

        # Collect all affected window indices
        affected = set()
        start_i = max(0, i - n + 1)
        end_i = min(i, L - n)
        for k in range(start_i, end_i + 1):
            affected.add(k)

        start_j = max(0, j - n + 1)
        end_j = min(j, L - n)
        for k in range(start_j, end_j + 1):
            affected.add(k)

        # Check if any affected window has count == 1
        for k in affected:
            if perm_count[tuple(word[k:k + n])] == 1:
                return False
        return True

    # 6. Helper: perform swap and update counts
    def do_swap(i, j):
        L = len(word)
        affected = set()
        start_i = max(0, i - n + 1)
        end_i = min(i, L - n)
        for k in range(start_i, end_i + 1):
            affected.add(k)
        start_j = max(0, j - n + 1)
        end_j = min(j, L - n)
        for k in range(start_j, end_j + 1):
            affected.add(k)

        old_vals = {k: tuple(word[k:k + n]) for k in affected}
        word[i], word[j] = word[j], word[i]

        for k in affected:
            old_p = old_vals[k]
            perm_count[old_p] -= 1
            new_p = tuple(word[k:k + n])
            perm_count[new_p] += 1

    # 7. Helper: check if move (from_i, to_i) is valid
    def can_move(from_i, to_i):
        if from_i == to_i:
            return False
        L = len(word)
        if L <= n:
            return False

        # Simulate the move: remove element at from_i, insert at to_i
        # Affected windows are those that include from_i or to_i (after removal)
        val = word[from_i]

        # Windows affected by removal at from_i
        affected_remove = set()
        start_r = max(0, from_i - n + 1)
        end_r = min(from_i, L - n)
        for k in range(start_r, end_r + 1):
            affected_remove.add(k)

        # After removal, the list has length L-1
        # The insertion index in the new list: if to_i > from_i, it's to_i - 1; else to_i
        new_to_i = to_i - 1 if to_i > from_i else to_i

        # Windows affected by insertion at new_to_i in the (L-1)-length list
        affected_insert = set()
        start_ins = max(0, new_to_i - n + 1)
        end_ins = min(new_to_i, L - 1 - n)  # new length is L-1, so last window starts at (L-1)-n
        for k in range(start_ins, end_ins + 1):
            affected_insert.add(k)

        all_affected = affected_remove | affected_insert

        # Check if any affected window currently has count == 1
        for k in all_affected:
            if perm_count[tuple(word[k:k + n])] == 1:
                return False
        return True

    # 8. Helper: perform move and update counts
    def do_move(from_i, to_i):
        L = len(word)
        val = word[from_i]

        # Affected windows by removal
        affected_remove = set()
        start_r = max(0, from_i - n + 1)
        end_r = min(from_i, L - n)
        for k in range(start_r, end_r + 1):
            affected_remove.add(k)

        # Decrement counts for removal-affected windows
        for k in affected_remove:
            p = tuple(word[k:k + n])
            perm_count[p] -= 1

        # Remove element
        word.pop(from_i)

        # Insertion index in new list
        new_to_i = to_i - 1 if to_i > from_i else to_i
        word.insert(new_to_i, val)

        # New length
        new_L = L
        # Affected windows by insertion
        affected_insert = set()
        start_ins = max(0, new_to_i - n + 1)
        end_ins = min(new_to_i, new_L - n)
        for k in range(start_ins, end_ins + 1):
            affected_insert.add(k)

        # Increment counts for insertion-affected windows
        for k in affected_insert:
            p = tuple(word[k:k + n])
            perm_count[p] += 1

    # 9. Aggressive deletion sweep: repeatedly delete any deletable position in a full pass
    def greedy_delete():
        progress = True
        while progress:
            progress = False
            L = len(word)
            i = 0
            while i < L:
                if can_delete(i):
                    delete(i)
                    progress = True
                    L = len(word)
                else:
                    i += 1
                if time.time() > deadline - 5:
                    return

    # 10. Try a single move operation, then greedy delete; return True if improved
    def try_improve():
        L_before = len(word)
        if L_before <= n + 1:
            return False

        op = random.random()
        improved = False

        if op < 0.5:
            # Delete
            idx = random.randint(0, L_before - 1)
            if can_delete(idx):
                delete(idx)
                greedy_delete()
                improved = len(word) < L_before
        elif op < 0.75:
            # Adjacent swap
            idx = random.randint(0, L_before - 2)
            if can_swap(idx, idx + 1):
                do_swap(idx, idx + 1)
                greedy_delete()
                improved = len(word) < L_before
                if not improved:
                    # Revert
                    do_swap(idx, idx + 1)
        else:
            # Move
            from_i = random.randint(0, L_before - 1)
            to_i = random.randint(0, L_before - 1)
            if from_i != to_i and can_move(from_i, to_i):
                do_move(from_i, to_i)
                greedy_delete()
                improved = len(word) < L_before
                if not improved:
                    # Revert: move back
                    # After the move, the element is at position:
                    # if to_i > from_i: it's at to_i - 1
                    # else: it's at to_i
                    # We want to move it back to from_i
                    if to_i > from_i:
                        new_pos = to_i - 1
                    else:
                        new_pos = to_i
                    do_move(new_pos, from_i)

        return improved

    # 11. Main search loop
    # Phase 1: Initial aggressive deletion
    greedy_delete()

    # Phase 2: Alternating between move-based improvement and deletion
    while time.time() < deadline - 3:
        if try_improve():
            continue
        # Occasionally do a pure deletion pass to clean up
        if random.random() < 0.2:
            greedy_delete()

    # Final cleanup
    greedy_delete()

    return "".join(map(str, word))
