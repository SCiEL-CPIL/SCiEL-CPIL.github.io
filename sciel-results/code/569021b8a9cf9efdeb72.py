<<<<<<< SEARCH
    def route_distance(route):
        """Return total distance of route (depot -> ... -> depot)."""
        if not route:
            return 0.0
        d = dist_mat[0, route[0]]
        for i in range(len(route) - 1):
            d += dist_mat[route[i], route[i + 1]]
        d += dist_mat[route[-1], 0]
        return d
    
    def check_feasibility(route):
        """Check if a route is feasible. Return (feasible, total_dist, total_time, max_battery_used)."""
        if not route:
            return True, 0.0, 0.0, 0.0
        
        t = 0.0
        bat = float(battery_max)
        total_d = 0.0
        total_demand = 0.0
        
        # Check capacity
        for c in route:
            total_demand += demands[c]
        if total_demand > capacity + 1e-9:
            return False, 0.0, 0.0, 0.0
        
        prev = 0
        for c in route:
            d = dist_mat[prev, c]
            t += d / speed
            bat -= d
            if bat < -1e-9:
                return False, 0.0, 0.0, 0.0
            
            # Time window
            if t < ready[c]:
                t = ready[c]
            if t > due[c] + 1e-9:
                return False, 0.0, 0.0, 0.0
            
            # Service
            t += service[c]
            
            # Charging? (We don't insert charging stations in the route list, 
            # but the problem says stations may be visited. For simplicity in heuristic,
            # we assume no intermediate charging unless needed. 
            # Actually, the route is a list of node ids. Stations are nodes. 
            # But we're only inserting customers. 
            # The problem says "stations may repeat" and a station adds time. 
            # For the heuristic, let's first solve without intermediate charging, 
            # and if infeasible due to battery, we'd need to insert charging stops.
            # Given battery=300 and distances in [0,100]^2, max leg ~141, 
            # a route of ~2-3 customers should be fine without charging.
            
            prev = c
        
        # Return to depot
        d = dist_mat[prev, 0]
        t += d / speed
        bat -= d
        if bat < -1e-9:
            return False, 0.0, 0.0, 0.0
        if t > horizon + 1e-9:
            return False, 0.0, 0.0, 0.0
        
        total_d += d
        return True, total_d, t, 0.0
=======
    def expand_route(customer_seq):
        """
        Takes a sequence of customer IDs and returns a full route (list of node IDs including stations)
        that is feasible, or None if infeasible.
        Also returns the total distance and total time.
        """
        if not customer_seq:
            return [], 0.0, 0.0
        
        total_demand = sum(demands[c] for c in customer_seq)
        if total_demand > capacity + 1e-9:
            return None, 0.0, 0.0

        full_route = []
        t = 0.0
        bat = float(battery_max)
        current_node = 0
        total_dist = 0.0
        
        # Helper to find best station to visit
        def find_best_station(cur, next_node):
            # We need to go from cur to next_node, but bat is insufficient.
            # We must visit a station.
            # We want to minimize the extra distance or time.
            # Let's pick the station that minimizes dist(cur, s) + dist(s, next_node).
            best_s = -1
            best_cost = float('inf')
            for s in stations:
                cost = dist_mat[cur, s] + dist_mat[s, next_node]
                if cost < best_cost:
                    best_cost = cost
                    best_s = s
            return best_s

        def try_move(cur, next_node):
            nonlocal t, bat, current_node, total_dist, full_route
            # If we can't reach next_node, find a station
            while dist_mat[cur, next_node] > bat + 1e-9:
                s = find_best_station(cur, next_node)
                if s == -1:
                    return False
                # Move to station
                d = dist_mat[cur, s]
                t += d / speed
                bat -= d
                total_dist += d
                if bat < -1e-9:
                    return False
                current_node = s
                full_route.append(s)
                
                # Charge
                charge_time = (battery_max - bat) * recharge_rate
                t += charge_time
                bat = battery_max
                
            # Move to next_node
            d = dist_mat[cur, next_node]
            t += d / speed
            bat -= d
            total_dist += d
            if bat < -1e-9:
                return False
            current_node = next_node
            full_route.append(next_node)
            return True

        # Visit customers
        for c in customer_seq:
            if not try_move(current_node, c):
                return None, 0.0, 0.0
            
            # Time window check
            if t < ready[c]:
                t = ready[c]
            if t > due[c] + 1e-9:
                return None, 0.0, 0.0
            
            t += service[c]

        # Return to depot
        if not try_move(current_node, 0):
            return None, 0.0, 0.0
        
        if t > horizon + 1e-9:
            return None, 0.0, 0.0
            
        return full_route, total_dist, t
>>>>>>> REPLACE

Now update the main logic to use `expand_route`.

