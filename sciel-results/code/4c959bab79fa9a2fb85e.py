import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin from 90s limit

    mask = (1 << 201) - 1

    def can_add(b, x):
        return ((b | (b << x)) >> 200) & 1 == 0

    def compute_bits(elements):
        b = 1
        for e in elements:
            b |= (b << e)
            b &= mask
        return b

    def verify(s):
        b = 1
        for e in s:
            b |= (b << e)
            b &= mask
        return ((b >> 200) & 1) == 0

    # Phase 1: Multiple greedy initializations
    best_set = set()
    best_size = 0

    # Try many random orders
    for trial in range(50):
        order = list(range(1, 101))
        random.shuffle(order)
        s = set()
        b = 1
        for x in order:
            if can_add(b, x):
                s.add(x)
                b = (b | (b << x)) & mask
        if len(s) > best_size:
            best_set = s
            best_size = len(s)

    # Also try sorted orders
    for order in [list(range(100, 0, -1)), list(range(1, 101)),
                  list(range(50, 0, -1)) + list(range(100, 50, -1))]:
        s = set()
        b = 1
        for x in order:
            if can_add(b, x):
                s.add(x)
                b = (b | (b << x)) & mask
        if len(s) > best_size:
            best_set = s
            best_size = len(s)

    s = set(best_set)
    b = compute_bits(list(s))
    size = len(s)
    global_best_set = set(s)
    global_best_size = size

    # Phase 2: Local search - try to add elements
    # Precompute which elements can be added
    def get_addable(s, b):
        addable = []
        for x in range(1, 101):
            if x not in s and can_add(b, x):
                addable.append(x)
        return addable

    improved = True
    while improved and (time.time() - start_time < DEADLINE):
        improved = False
        addable = get_addable(s, b)
        if addable:
            # Try each addable element, keep the one that allows most future additions
            best_x = None
            best_future = -1
            for x in addable:
                ns = s | {x}
                nb = (b | (b << x)) & mask
                future = len(get_addable(ns, nb))
                if future > best_future:
                    best_future = future
                    best_x = x
            if best_x is not None:
                s = s | {best_x}
                b = (b | (b << best_x)) & mask
                size = len(s)
                improved = True
                if size > global_best_size:
                    global_best_set = set(s)
                    global_best_size = size
        else:
            # No simple addition works, try a swap: remove one, add one
            elements = list(s)
            random.shuffle(elements)
            candidates = [x for x in range(1, 101) if x not in s]
            random.shuffle(candidates)
            found_swap = False
            for y in elements:
                if found_swap:
                    break
                s_minus = s - {y}
                b_minus = compute_bits(list(s_minus))
                for x in candidates:
                    if can_add(b_minus, x):
                        ns = s_minus | {x}
                        nb = (b_minus | (b_minus << x)) & mask
                        # Check if we can add more after this swap
                        addable_after = get_addable(ns, nb)
                        if addable_after:
                            s = ns
                            b = nb
                            size = len(s)
                            found_swap = True
                            improved = True
                            if size > global_best_size:
                                global_best_set = set(s)
                                global_best_size = size
                            break
            if not found_swap:
                # Try double swap: remove 2, add 2
                n = len(elements)
                for i in range(n):
                    if found_swap or time.time() - start_time > DEADLINE:
                        break
                    for j in range(i + 1, n):
                        if found_swap or time.time() - start_time > DEADLINE:
                            break
                        y1, y2 = elements[i], elements[j]
                        s_minus = s - {y1, y2}
                        b_minus = compute_bits(list(s_minus))
                        for x in candidates:
                            if found_swap:
                                break
                            if can_add(b_minus, x):
                                b1 = (b_minus | (b_minus << x)) & mask
                                s1 = s_minus | {x}
                                for x2 in candidates:
                                    if found_swap:
                                        break
                                    if x2 != x and can_add(b1, x2):
                                        b2 = (b1 | (b1 << x2)) & mask
                                        s2 = s1 | {x2}
                                        addable_after = get_addable(s2, b2)
                                        if addable_after:
                                            s = s2
                                            b = b2
                                            size = len(s)
                                            found_swap = True
                                            improved = True
                                            if size > global_best_size:
                                                global_best_set = set(s)
                                                global_best_size = size
                                            break

    # Final verification
    if not verify(global_best_set):
        if verify(best_set):
            return list(best_set)
        return [100]

    return list(global_best_set)
