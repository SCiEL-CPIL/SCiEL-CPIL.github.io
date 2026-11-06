import time

def _all_products(values):
    products = set()
    for i in range(len(values)):
        p = 1
        for j in range(i, len(values)):
            p *= values[j]
            products.add(p)
    return products

def _can_insert_at(values, products, x, pos):
    if values:
        if pos > 0 and values[pos-1] >= x:
            return False
        if pos < len(values) and values[pos] <= x:
            return False
    left_prods = [1]
    p = 1
    for i in range(pos-1, -1, -1):
        p *= values[i]
        left_prods.append(p)
    right_prods = [1]
    p = 1
    for j in range(pos, len(values)):
        p *= values[j]
        right_prods.append(p)
    for lp in left_prods:
        for rp in right_prods:
            if lp * x * rp in products:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    def time_left():
        return deadline - time.time()

    # Phase 1: Greedy
    values, products = [], set()
    for x in range(2, 129):
        if time_left() < 1.0:
            break
        tails = [x]
        p = x
        for a in reversed(values):
            p *= a
            tails.append(p)
        if not (products & set(tails)):
            values.append(x)
            products.update(tails)

    best_values = values[:]
    best_len = len(values)

    # Phase 2: Repeated best insertion
    for _ in range(200):
        if time_left() < 1.0:
            break
        excluded = [x for x in range(2, 129) if x not in set(values)]
        best_move = None
        for x in excluded:
            if time_left() < 0.5:
                break
            for pos in range(len(values) + 1):
                if values and (pos > 0 and values[pos-1] >= x or pos < len(values) and values[pos] <= x):
                    continue
                if _can_insert_at(values, products, x, pos):
                    if best_move is None:
                        best_move = (x, pos)
                    break
        if best_move:
            x, pos = best_move
            values.insert(pos, x)
            products = _all_products(values)
            if len(values) > best_len:
                best_len = len(values)
                best_values = values[:]
        else:
            break

    # Phase 3: Local search - remove one, insert another
    if time_left() > 3.0:
        improved = True
        while improved and time_left() > 1.0:
            improved = False
            cur_set = set(values)
            excluded = [x for x in range(2, 129) if x not in cur_set]
            for remove_idx in range(len(values)):
                if time_left() < 0.5:
                    break
                temp_values = values[:remove_idx] + values[remove_idx+1:]
                temp_products = _all_products(temp_values)
                for x in excluded:
                    if time_left() < 0.3:
                        break
                    for pos in range(len(temp_values) + 1):
                        if temp_values and (pos > 0 and temp_values[pos-1] >= x or pos < len(temp_values) and temp_values[pos] <= x):
                            continue
                        if _can_insert_at(temp_values, temp_products, x, pos):
                            new_values = temp_values[:]
                            new_values.insert(pos, x)
                            if len(new_values) > best_len:
                                values = new_values
                                products = _all_products(values)
                                best_len = len(values)
                                best_values = values[:]
                                improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

    return best_values
