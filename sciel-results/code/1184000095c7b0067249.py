"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 256

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 25.0
    best_size = N + 1
    best_set = None

    def _can_add(x, selected_set, sums_mask):
        """Check if x can be added to the Sidon set. selected_set is a set of ints."""
        if x in selected_set:
            return False
        if (sums_mask >> (2 * x)) & 1:
            return False
        for a in selected_set:
            if (sums_mask >> (x + a)) & 1:
                return False
        return True

    def _get_addable(selected_set, sums_mask):
        """Get list of all elements that can be added."""
        addable = []
        for x in range(1, N + 1):
            if x in selected_set:
                continue
            if (sums_mask >> (2 * x)) & 1:
                continue
            ok = True
            for a in selected_set:
                if (sums_mask >> (x + a)) & 1:
                    ok = False
                    break
            if ok:
                addable.append(x)
        return addable

    def _score_element(x, selected_set, sums_mask, addable):
        """Score element x by how many other addable elements it would block."""
        new_sums = sums_mask | (1 << (2 * x))
        for a in selected_set:
            new_sums |= 1 << (x + a)
        
        blocked = 0
        for y in addable:
            if y == x:
                continue
            if (new_sums >> (2 * y)) & 1:
                blocked += 1
                continue
            blocked_y = False
            for a in selected_set:
                if (new_sums >> (y + a)) & 1:
                    blocked_y = True
                    break
            if not blocked_y and (new_sums >> (y + x)) & 1:
                blocked_y = True
            if blocked_y:
                blocked += 1
        return blocked

    def _greedy_construct(order_seed):
        """Greedy construction with a given element ordering."""
        rng = random.Random(order_seed)
        remaining = list(range(1, N + 1))
        rng.shuffle(remaining)
        selected = []
        selected_set = set()
        sums_mask = 0
        while remaining:
            best_x = None
            best_block = -1
            for x in remaining:
                if x in selected_set:
                    continue
                if (sums_mask >> (2 * x)) & 1:
                    continue
                ok = True
                for a in selected_set:
                    if (sums_mask >> (x + a)) & 1:
                        ok = False
                        break
                if not ok:
                    continue
                new_sums = sums_mask | (1 << (2 * x))
                for a in selected_set:
                    new_sums |= 1 << (x + a)
                blocked = 0
                for y in remaining:
                    if y == x or y in selected_set:
                        continue
                    if (new_sums >> (2 * y)) & 1:
                        blocked += 1
                        continue
                    blocked_y = False
                    for a in selected_set:
                        if (new_sums >> (y + a)) & 1:
                            blocked_y = True
                            break
                    if not blocked_y and (new_sums >> (y + x)) & 1:
                        blocked_y = True
                    if blocked_y:
                        blocked += 1
                if blocked > best_block:
                    best_block = blocked
                    best_x = x
            if best_x is None:
                break
            selected.append(best_x)
            selected_set.add(best_x)
            sums_mask |= 1 << (2 * best_x)
            for a in selected_set:
                if a != best_x:
                    sums_mask |= 1 << (best_x + a)
            remaining.remove(best_x)
        return selected

    def _backtrack(selected, selected_set, sums_mask):
        nonlocal best_size, best_set
        if time.time() > deadline:
            return

        addable = _get_addable(selected_set, sums_mask)

        if not addable:
            if len(selected) < best_size:
                best_size = len(selected)
                best_set = list(selected)
            return

        if len(selected) >= best_size:
            return

        scored = []
        for x in addable:
            blocked = _score_element(x, selected_set, sums_mask, addable)
            scored.append((blocked, x))

        scored.sort(key=lambda t: (-t[0], t[1]))

        for _, x in scored[:4]:
            new_sums = sums_mask | (1 << (2 * x))
            for a in selected_set:
                new_sums |= 1 << (x + a)
            
            selected.append(x)
            selected_set.add(x)
            _backtrack(selected, selected_set, new_sums)
            selected.pop()
            selected_set.discard(x)
            
            if time.time() > deadline:
                return

    def _local_search_polish(sel):
        """Try to remove elements from sel while maintaining maximality."""
        if time.time() > deadline:
            return sel
        current = list(sel)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_set = set(current)
            sums_mask = 0
            for a in current:
                sums_mask |= 1 << (2 * a)
                for b in current:
                    if b > a:
                        sums_mask |= 1 << (a + b)
            
            for i in range(len(current)):
                if time.time() > deadline:
                    return current
                x = current[i]
                # Try removing x
                trial = [e for e in current if e != x]
                trial_set = set(trial)
                
                # Check if trial is still Sidon
                trial_sums = 0
                valid = True
                for a in trial:
                    if (trial_sums >> (2 * a)) & 1:
                        valid = False
                        break
                    trial_sums |= 1 << (2 * a)
                    for b in trial:
                        if b >= a:
                            if (trial_sums >> (a + b)) & 1:
                                valid = False
                                break
                            trial_sums |= 1 << (a + b)
                    if not valid:
                        break
                if not valid:
                    continue
                
                # Check if trial is still maximal
                is_maximal = True
                for y in range(1, N + 1):
                    if y in trial_set:
                        continue
                    if (trial_sums >> (2 * y)) & 1:
                        continue
                    ok = True
                    for a in trial_set:
                        if (trial_sums >> (y + a)) & 1:
                            ok = False
                            break
                    if ok:
                        is_maximal = False
                        break
                if is_maximal:
                    current = trial
                    improved = True
                    break
        return current

    # Run greedy with multiple seeds
    random.seed(42)
    for seed in range(10):
        if time.time() > deadline:
            break
        sel = _greedy_construct(seed)
        sel = _local_search_polish(sel)
        if len(sel) < best_size:
            best_size = len(sel)
            best_set = list(sel)

    # Backtracking search
    for restart in range(8):
        if time.time() > deadline:
            break
        selected_bt = []
        selected_set_bt = set()
        _backtrack(selected_bt, selected_set_bt, 0)

    if best_set is not None:
        return best_set
    else:
        return _greedy_construct(0)
# EVOLVE-BLOCK-END
