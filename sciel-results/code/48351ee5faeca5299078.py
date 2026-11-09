    def _solve_instance(instance, time_limit=45):
        start_time = time.time()
        deadline = start_time + time_limit

        # Create a local copy to safely modify
        instance = dict(instance)
        
        coords = instance["coords"]
        n_nodes = len(coords)
        
        def to_list(data):
            if isinstance(data, list):
                if len(data) >= n_nodes:
                    return data
                return data + [0] * (n_nodes - len(data))
            elif isinstance(data, dict):
                lst = [0] * n_nodes
                for k, v in data.items():
                    try:
                        idx = int(k)
                    except (ValueError, TypeError):
                        continue
                    if 0 <= idx < n_nodes:
                        lst[idx] = v
                return lst
            else:
                return [0] * n_nodes

        instance["demands"] = to_list(instance.get("demands", {}))
        instance["ready"] = to_list(instance.get("ready", {}))
        instance["due"] = to_list(instance.get("due", {}))
        instance["service"] = to_list(instance.get("service", {}))

        dist = _dist_matrix(coords)

        battery_max = instance["battery"]
        capacity = instance["capacity"]
        speed = instance["speed"]
        recharge_rate = instance["recharge_rate"]
        horizon = instance["horizon"]
        n_customers = instance["n_customers"]
        demands = instance["demands"]
        ready = instance["ready"]
        due = instance["due"]
        service = instance["service"]
        stations = set(instance["stations"])
    