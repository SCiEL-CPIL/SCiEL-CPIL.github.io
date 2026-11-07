<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80 second deadline, 10s safety margin
    
    all_edges = _all_edges(P['n'])
    
    best_score = float('inf')
    best_edges = None
    best_values = None
    
    value_strategies = [
        _make_value_set_1,
        _make_value_set_2,
        _make_value_set_3,
        _make_value_set_4,
        _make_value_set_5,
        _make_value_set_6,
    ]
    
    for make_values in value_strategies:
        if time.time() > deadline - 15:
            break
        
        values = make_values()
        if len(set(values)) != 14:
            continue
        if not all(isinstance(v, int) and 0 < v <= 10000 for v in values):
            continue
        
        score, edges = _optimize_values(values, all_edges, deadline)
        if edges is not None and score < best_score:
            best_score = score
            best_edges = edges
            best_values = values[:]
    
    # Fallback
    if best_edges is None:
        values1 = _make_value_set_1()
        edges = _greedy_edge_selection(values1, P['m'], all_edges)
        best_values = values1
        best_edges = edges
        best_score = _raw_score(best_values, best_edges)
    
    # Ensure types are correct
    final_values = [int(v) for v in best_values]
    final_edges = [[int(u), int(v)] for u, v in best_edges]
    
    return {'values': final_values, 'edges': final_edges}
=======
def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75 second deadline, 15s safety margin
    
    all_edges = _all_edges(P['n'])
    
    best_score = float('inf')
    best_edges = None
    best_values = None
    
    value_strategies = [
        _make_value_set_1,
        _make_value_set_2,
        _make_value_set_3,
        _make_value_set_4,
        _make_value_set_5,
        _make_value_set_6,
        _make_value_set_7,
        _make_value_set_8,
    ]
    
    for make_values in value_strategies:
        if time.time() > deadline - 12:
            break
        
        values = make_values()
        if len(set(values)) != 14:
            continue
        if not all(isinstance(v, int) and 0 < v <= 10000 for v in values):
            continue
        
        score, edges, vals = _optimize_values(values, all_edges, deadline)
        if edges is not None and score < best_score:
            best_score = score
            best_edges = edges
            best_values = vals if vals is not None else values[:]
    
    # Fallback
    if best_edges is None:
        values1 = _make_value_set_1()
        edges = _greedy_edge_selection(values1, P['m'], all_edges)
        best_values = values1
        best_edges = edges
        best_score = _raw_score(best_values, best_edges)
    
    # Ensure types are correct
    final_values = [int(v) for v in best_values]
    final_edges = [[int(u), int(v)] for u, v in best_edges]
    
    return {'values': final_values, 'edges': final_edges}
>>>>>>> REPLACE
