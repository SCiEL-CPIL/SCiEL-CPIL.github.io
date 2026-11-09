<<<<<<< SEARCH
def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    deadline = time.time() + 115  # Safety margin
    
    if n == 0:
        return []
    
    heuristic_orders = _heuristic_orders(boxes, n)
    
    best_result = None
    best_vol = -1
    
    # Run skyline greedy with different orderings
    for name, order in heuristic_orders:
        if time.time() > deadline - 30:
            break
        result = _greedy_skyline(order, boxes, container, deadline)
        vol = _compute_volume(result)
        if vol > best_vol:
            best_vol = vol
            best_result = result
    
    # Run shelf greedy with different orderings
    if time.time() < deadline - 20:
        for name, order in heuristic_orders[:4]:
            if time.time() > deadline - 15:
                break
            result = _greedy_shelf(order, boxes, container, deadline)
            vol = _compute_volume(result)
            if vol > best_vol:
                best_vol = vol
                best_result = result
    
    # Random restarts with skyline
    if time.time() < deadline - 15 and best_result is not None:
        rng_restart = random.Random(123)
        for _ in range(8):
            if time.time() > deadline - 10:
                break
            random_order = list(range(n))
            rng_restart.shuffle(random_order)
            result_rand = _greedy_skyline(random_order, boxes, container, deadline)
            vol_rand = _compute_volume(result_rand)
            if vol_rand > best_vol:
                best_vol = vol_rand
                best_result = result_rand
            # Also try a hybrid order: volume desc but with some randomization
            if time.time() < deadline - 10:
                hybrid_order = list(range(n))
                rng_restart.shuffle(hybrid_order)
                hybrid_order.sort(key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2]*0.7 + rng_restart.random()*0.3*max(b[0]*b[1]*b[2] for b in boxes), reverse=True)
                result_hybrid = _greedy_skyline(hybrid_order, boxes, container, deadline)
                vol_hybrid = _compute_volume(result_hybrid)
                if vol_hybrid > best_vol:
                    best_vol = vol_hybrid
                    best_result = result_hybrid
    
    # Local search to improve
    remaining_time = deadline - time.time()
    if remaining_time > 5 and best_result is not None:
        max_iters = int(remaining_time * 30)
        max_iters = max(100, min(max_iters, 1500))
        best_result = _local_search(best_result, boxes, container, deadline, max_iters=max_iters)
    
    # Final multi-start polish: try a few more greedy passes with different seeds
    if best_result is not None and time.time() < deadline - 5:
        rng_final = random.Random(999)
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            random_order = list(range(n))
            rng_final.shuffle(random_order)
            result_final = _greedy_skyline(random_order, boxes, container, deadline)
            vol_final = _compute_volume(result_final)
            if vol_final > best_vol:
                best_vol = vol_final
                best_result = result_final
                # Run local search on the improved result
                if time.time() < deadline - 3:
                    max_iters2 = max(50, int((deadline - time.time() - 3) * 20))
                    best_result = _local_search(best_result, boxes, container, deadline, max_iters=max_iters2)
                    best_vol = _compute_volume(best_result)
    
    if best_result is None:
        best_result = []
        for w, h, d in boxes:
            best_result.append((-1, 0, 0, 0, w, h, d))
    
    return [tuple(r) for r in best_result]
=======
def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    deadline = time.time() + 115  # Safety margin
    
    if n == 0:
        return []
    
    heuristic_orders = _heuristic_orders(boxes, n)
    
    best_result = None
    best_vol = -1
    
    # Run skyline greedy with different orderings (top 4 most promising)
    for name, order in heuristic_orders[:4]:
        if time.time() > deadline - 40:
            break
        result = _greedy_skyline(order, boxes, container, deadline)
        vol = _compute_volume(result)
        if vol > best_vol:
            best_vol = vol
            best_result = result
    
    # Random restarts with skyline (more diverse)
    if time.time() < deadline - 20 and best_result is not None:
        rng_restart = random.Random(123)
        for _ in range(6):
            if time.time() > deadline - 15:
                break
            random_order = list(range(n))
            rng_restart.shuffle(random_order)
            result_rand = _greedy_skyline(random_order, boxes, container, deadline)
            vol_rand = _compute_volume(result_rand)
            if vol_rand > best_vol:
                best_vol = vol_rand
                best_result = result_rand
    
    # Local search to improve
    remaining_time = deadline - time.time()
    if remaining_time > 10 and best_result is not None:
        max_iters = int(remaining_time * 40)
        max_iters = max(200, min(max_iters, 3000))
        best_result = _local_search(best_result, boxes, container, deadline, max_iters=max_iters)
    
    # Final multi-start polish
    if best_result is not None and time.time() < deadline - 5:
        rng_final = random.Random(999)
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            random_order = list(range(n))
            rng_final.shuffle(random_order)
            result_final = _greedy_skyline(random_order, boxes, container, deadline)
            vol_final = _compute_volume(result_final)
            if vol_final > best_vol:
                best_vol = vol_final
                best_result = result_final
                if time.time() < deadline - 3:
                    max_iters2 = max(100, int((deadline - time.time() - 3) * 30))
                    best_result = _local_search(best_result, boxes, container, deadline, max_iters=max_iters2)
                    best_vol = _compute_volume(best_result)
    
    if best_result is None:
        best_result = []
        for w, h, d in boxes:
            best_result.append((-1, 0, 0, 0, w, h, d))
    
    return [tuple(r) for r in best_result]
>>>>>>> REPLACE
