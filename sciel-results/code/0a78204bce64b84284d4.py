def dispatch(ready_operations, time):
       if not ready_operations:
           return 0
       
       best_idx = -1
       best_score = float('inf')
       best_machine = None
       
       # Pre-compute machine competition?
       # For each machine, how many ready ops want it?
       machine_demand = {}
       for op in ready_operations:
           for m in op["machine_options"]:
               machine_demand[m] = machine_demand.get(m, 0) + 1
       
       for i, op in enumerate(ready_operations):
           # Find the best machine for this op
           # Criteria:
           # 1. Shortest duration (SPT)
           # 2. Machine with lowest demand (least contention)
           # 3. Machine with lowest historical load (if we track it)
           
           best_m_for_op = None
           best_d_for_op = float('inf')
           best_demand_for_op = float('inf')
           
           for m, dur in op["machine_options"].items():
               demand = machine_demand.get(m, 0)
               # Score for this machine choice:
               # We want low duration and low demand.
               # Combine them.
               # Let's say primary is duration, secondary is demand.
               
               if dur < best_d_for_op or (dur == best_d_for_op and demand < best_demand_for_op):
                   best_d_for_op = dur
                   best_m_for_op = m
                   best_demand_for_op = demand
           
           # Now, score the operation itself
           # We want to pick the op with the smallest best_d_for_op
           # Tie-breaker: Job with most remaining work? Or least?
           # Usually, finishing a job (least remaining work) is good to clear the board.
           # But SPT suggests short duration is good.
           
           # Let's use a composite score where lower is better.
           # Score = best_d_for_op + alpha * best_demand_for_op
           score = best_d_for_op + 0.1 * best_demand_for_op
           
           if score < best_score:
               best_score = score
               best_idx = i
               best_machine = best_m_for_op
               
       return (best_idx, best_machine)
