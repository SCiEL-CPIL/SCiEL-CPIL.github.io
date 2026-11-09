<<<<<<< SEARCH
def _simulate_route_v2(route, D, ready, due, service, demands,
                       battery, speed, recharge_rate, horizon, capacity, n_nodes, stations_set):
    """Proper simulation with charging stations."""
    total_dist = 0.0
    t = 0.0
    bat = float(battery)
    prev = 0
    for node in route:
        d = D[prev, node]
        total_dist += d
        t += d / speed
        bat -= d
        if bat < -1e-9:
            return False, total_dist, t, bat
        bat = max(bat, 0.0)
        if node in stations_set:
            # Charging station: recharge time = (max_battery - arrival_battery) * recharge_rate
            recharge_time = (float(battery) - bat) * recharge_rate
            t += recharge_time
            bat = float(battery)
        elif node >= 1 and node < n_nodes and demands[node] > 0:
            # Customer
            t = max(t, ready[node])
            if t > due[node] + 1e-9:
                return False, total_dist, t, bat
            t += service[node]
        prev = node
    # return to depot
    d = D[prev, 0]
    total_dist += d
    t += d / speed
    bat -= d
    if bat < -1e-9:
        return False, total_dist, t, bat
    if t > horizon + 1e-9:
        return False, total_dist, t, bat
    return True, total_dist, t, bat
=======
def _simulate_route_v2(route, D, ready, due, service, demands,
                       battery, speed, recharge_rate, horizon, capacity, n_nodes, stations_set):
    """Proper simulation with charging stations."""
    total_dist = 0.0
    t = 0.0
    bat = float(battery)
    prev = 0
    for node in route:
        d = D[prev, node]
        total_dist += d
        t += d / speed
        bat -= d
        if bat < -1e-9:
            return False, total_dist, t, bat
        # Clamp to 0 for recharge calculation, but remember arrival state
        arrival_bat = max(bat, 0.0)
        
        if node in stations_set:
            # Charging station: recharge time = (max_battery - arrival_battery) * recharge_rate
            recharge_time = (float(battery) - arrival_bat) * recharge_rate
            t += recharge_time
            bat = float(battery)
        elif node >= 1 and node < n_nodes and demands[node] > 0:
            # Customer
            t = max(t, ready[node])
            if t > due[node] + 1e-9:
                return False, total_dist, t, bat
            t += service[node]
        prev = node
    # return to depot
    d = D[prev, 0]
    total_dist += d
    t += d / speed
    bat -= d
    if bat < -1e-9:
        return False, total_dist, t, bat
    if t > horizon + 1e-9:
        return False, total_dist, t, bat
    return True, total_dist, t, bat
>>>>>>> REPLACE

