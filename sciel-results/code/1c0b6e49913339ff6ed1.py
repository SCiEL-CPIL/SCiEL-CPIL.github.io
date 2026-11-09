"""EV routing via precomputed shortest paths and cluster-based route construction."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import permutations


def _dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def plan(instance):
    coords = np.asarray(instance["coords"], dtype=float)
    n = instance["n_customers"]
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = set(instance["stations"])
    depot = 0

    # Precompute all pairwise distances
    n_total = len(coords)
    dist = np.zeros((n_total, n_total))
    for i in range(n_total):
        for j in range(n_total):
            dist[i][j] = _dist(coords[i], coords[j])

    remaining = list(range(1, n + 1))
    # Sort by due date for priority
    remaining.sort(key=lambda v: (due[v], v))

    routes = []

    def find_best_station_insertion(u, v, current_batt, current_time, current_load):
        """
        Given we are at u and want to go to v, but battery is insufficient.
        Try inserting a station s between u and v.
        Returns (station_id, new_batt_after_v, new_time_after_v, extra_dist) if feasible, else None.
        Note: This function assumes we are checking feasibility of the leg u->v.
        """
        best_s = -1
        best_cost = float('inf')
        best_batt_v = 0
        best_time_v = 0
        best_dist_add = 0
        
        d_uv = dist[u][v]
        
        for s in stations:
            if s == u or s == v:
                continue
            d_us = dist[u][s]
            d_sv = dist[s][v]
            
            # Check if u->s is feasible with current_batt
            if current_batt - d_us < -1e-9:
                continue
            
            # Time to reach s
            t_s = current_time + d_us / speed
            # Recharge at s
            # Battery at s arrival is current_batt - d_us
            # Recharge time = (battery_capacity - batt_arrival) * recharge_rate
            # Wait, the rule says: "a station adds (battery - energy) * recharge_rate time and refills"
            # "refills" implies battery becomes full (capacity? or just 'battery' limit?)
            # Task: "battery 300". "refills". Usually means to max capacity.
            # Let's assume it refills to `battery` (max capacity).
            
            batt_arr_s = current_batt - d_us
            recharge_time = (battery - batt_arr_s) * recharge_rate
            t_leave_s = t_s + recharge_time
            batt_leave_s = float(battery) # Refilled
            
            # Check if s->v is feasible with full battery
            if batt_leave_s - d_sv < -1e-9:
                continue
            
            t_arr_v = t_leave_s + d_sv / speed
            batt_arr_v = batt_leave_s - d_sv
            
            # Check time window for v
            t_v = max(t_arr_v, ready[v])
            if t_v > due[v] + 1e-9:
                continue
            
            # If feasible, calculate cost
            # Cost is primarily distance, but we also care about time
            # For insertion heuristic, we usually minimize distance.
            extra_dist = d_us + d_sv - d_uv # The detour
            # We want to minimize extra_dist, and maybe recharge_time
            cost = extra_dist + 0.1 * recharge_time
            
            if cost < best_cost:
                best_cost = cost
                best_s = s
                best_batt_v = batt_arr_v
                best_time_v = t_v + service[v]
                best_dist_add = d_us + d_sv

        if best_s != -1:
            return best_s, best_batt_v, best_time_v, best_dist_add
        return None

    def check_feasible_sequence(seq):
        """
        Check if a sequence of customers is feasible.
        Returns (total_distance, final_time, full_node_sequence) if feasible, else None.
        full_node_sequence includes stations if needed.
        """
        cur = depot
        time = 0.0
        batt = float(battery)
        load = 0
        total_d = 0.0
        full_seq = [] # To track actual nodes visited
        
        for v in seq:
            d = dist[cur][v]
            t_arr = time + d / speed
            b_arr = batt - d
            
            if b_arr < -1e-9:
                # Try to insert a station
                result = find_best_station_insertion(cur, v, batt, time, load)
                if result is None:
                    return None
                s, new_batt_v, new_time_v, dist_add = result
                # Update state to be at v after visiting s
                # We need to update total_d and full_seq
                total_d += dist_add
                full_seq.append(s)
                full_seq.append(v)
                cur = v
                batt = new_batt_v
                time = new_time_v
                load += demands[v]
                if load > capacity + 1e-9:
                    return None
                continue
            else:
                # Direct travel is feasible
                t = max(t_arr, ready[v])
                if t > due[v] + 1e-9:
                    return None
                t_serv = t + service[v]
                load += demands[v]
                if load > capacity + 1e-9:
                    return None
                time = t_serv
                batt = b_arr
                total_d += d
                full_seq.append(v)
                cur = v
        
        # Return to depot
        d_ret = dist[cur][depot]
        b_ret = batt - d_ret
        if b_ret < -1e-9:
             # Try to insert station on return leg?
             # This is harder. For now, if we can't return directly, we might need a station.
             # Let's check if we can go to a station and then depot.
             found = False
             for s in stations:
                 d_cs = dist[cur][s]
                 if batt - d_cs < -1e-9:
                     continue
                 t_s = time + d_cs / speed
                 batt_arr_s = batt - d_cs
                 recharge_time = (battery - batt_arr_s) * recharge_rate
                 t_leave_s = t_s + recharge_time
                 batt_leave_s = float(battery)
                 d_sd = dist[s][depot]
                 if batt_leave_s - d_sd < -1e-9:
                     continue
                 t_d = t_leave_s + d_sd / speed
                 if t_d > horizon + 1e-9:
                     continue
                 # Feasible via station s
                 total_d += d_cs + d_sd
                 full_seq.append(s)
                 # Note: we don't add depot to full_seq as it's implicit
                 return (total_d, t_d, full_seq)
             # If no station works, infeasible
             return None
        else:
            t_ret = time + d_ret / speed
            if t_ret > horizon + 1e-9:
                return None
            total_d += d_ret
            return (total_d, t_ret, full_seq)

    def try_insert(customer, seq):
        """Try to insert customer into seq at best position. Returns (new_seq, full_node_seq) or None."""
        best_pos = -1
        best_cost = float('inf')
        best_full_seq = None
        
        for i in range(len(seq) + 1):
            new_seq = seq[:i] + [customer] + seq[i:]
            result = check_feasible_sequence(new_seq)
            if result is not None:
                total_d, final_time, full_node_seq = result
                if total_d < best_cost:
                    best_cost = total_d
                    best_pos = i
                    best_full_seq = full_node_seq
        
        if best_pos >= 0:
            return seq[:best_pos] + [customer] + seq[best_pos:], best_full_seq
        return None

    # Build routes using insertion heuristic
    while remaining:
        route_customers = []
        route_full_nodes = []
        
        # Start with the customer with earliest due date
        start = remaining[0]
        route_customers.append(start)
        remaining.remove(start)
        
        # Initialize full nodes for the first customer
        # We need to check if depot->start is feasible directly or via station
        # check_feasible_sequence handles this, but it expects a seq.
        # Let's just use the check function for the initial route.
        init_result = check_feasible_sequence(route_customers)
        if init_result is None:
             # Should not happen for single customer unless depot->cust is infeasible
             # which is unlikely given battery=300 and max dist=141
             pass
        else:
            route_full_nodes = init_result[2]

        # Try to insert other customers
        changed = True
        while changed:
            changed = False
            for v in list(remaining):
                res = try_insert(v, route_customers)
                if res is not None:
                    new_cust_seq, new_full_seq = res
                    route_customers = new_cust_seq
                    route_full_nodes = new_full_seq
                    remaining.remove(v)
                    changed = True
        
        if route_customers:
            routes.append(route_full_nodes)

    return routes
# EVOLVE-BLOCK-END
