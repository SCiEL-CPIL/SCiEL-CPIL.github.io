"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

# Deadline guard
_START_TIME = None
_DEADLINE = 145.0

# State for load tracking
_machine_loads = {}
_last_dispatch_time = 0.0

def dispatch(ready_operations, time):
    global _START_TIME
    global _machine_loads
    global _last_dispatch_time
    
    if _START_TIME is None:
        _START_TIME = _time.time()
    
    # Deadline check
    elapsed = _time.time() - _START_TIME
    if elapsed > _DEADLINE - 5.0:
        # Fallback: MWR (Most Work Remaining)
        if not ready_operations:
            return 0
        return int(max(range(len(ready_operations)), key=lambda i: ready_operations[i].get("remaining_work", 0)))
    
    try:
        if not ready_operations:
            return 0
        
        n = len(ready_operations)
        
        if n == 1:
            # If only one op, we still might want to specify the best machine if we can
            # But for safety and simplicity in single-op case, just return index 0.
            # The simulator will pick a machine or we can return tuple.
            # Let's return tuple to be consistent if we can pick.
            op = ready_operations[0]
            mo = op.get("machine_options", {})
            if mo:
                # Pick machine with lowest current load
                best_m = min(mo.keys(), key=lambda m: _machine_loads.get(m, 0))
                return (0, best_m)
            return 0
        
        # 1. Update machine loads based on time elapsed since last dispatch
        # We assume the simulator advances time to the next event.
        # If we dispatched at _last_dispatch_time and now it is 'time',
        # machines that were busy might be free, but we don't know which ones were busy.
        # However, 'machine_options' in the current call only lists machines available NOW.
        # If a machine is in 'machine_options', it is free at 'time'.
        # Therefore, any machine NOT in the union of all machine_options in this call might be busy.
        # But we can't easily track "who is busy" without knowing the duration of the last op.
        
        # Alternative Load Estimation Strategy:
        # Since we don't have full history, we estimate load by "recent usage frequency" 
        # or simply by assuming that machines with shorter durations in the current 
        # ready set are less likely to be bottlenecks if we pick them, but we want 
        # to balance load.
        
        # A simpler, robust heuristic for flexible job shop without full state:
        # Minimize: (Job Criticality) * (Machine Load Estimate)
        
        # Let's refine the "Load Estimate".
        # If we pick a machine, its load increases by the duration.
        # We want to avoid picking a machine that is likely to be congested.
        # Without knowing who is busy, we can use a "decay" model.
        # But the most effective structural change is to look at the *duration* itself.
        # If we pick the shortest duration on a machine, we free it up sooner.
        # This is "Shortest Processing Time" (SPT) for the specific machine.
        
        # However, the previous insight mentioned "greedy SPT-based load simulation".
        # Let's implement a lightweight simulation of the next few steps? No, too slow.
        
        # Let's use a composite score:
        # Score(i, m) = remaining_work_i * (1 + load_penalty * current_load_estimate_m)
        # We want to MINIMIZE this score? No, we want to prioritize high remaining work.
        # Actually, standard dispatching rules for makespan often use:
        # 1. Critical Ratio (CR) = Remaining Time / Remaining Work. Lower is better?
        # 2. Minimum Slack.
        
        # Let's try a different structural approach: **Greedy Load Balancing**.
        # We maintain a dictionary of estimated loads.
        # When we dispatch an op to a machine, we add its duration to that machine's load.
        # We subtract a decay factor or reset loads when we see a machine is available again?
        # If a machine is in 'machine_options', it is available.
        # If we assume the simulator is event-driven, and we just dispatched at T_last,
        # and now it is T_now, any machine that was busy until T_now is now free.
        # We don't know which ones were busy.
        
        # Let's use a "Round Robin" style tie-breaker combined with Criticality.
        # But the directive asks for a structural change.
        
        # New Strategy: **Minimize Weighted Completion Time**
        # For each ready op i and each eligible machine m:
        #   est_completion = time + duration(i, m)
        #   job_weight = remaining_work(i)
        #   machine_congestion = _machine_loads.get(m, 0)
        #   
        #   # We want to pick the op/machine that results in the "best" system state.
        #   # A common proxy for makespan improvement is minimizing the sum of weighted completion times.
        #   # But here we are just picking ONE op to start.
        #   # So we should pick the op that has the highest "urgency" but on a machine that is "least loaded".
        
        # Let's define Urgency(i) = remaining_work(i)
        # Let's define Load(m) = _machine_loads.get(m, 0)
        
        # We want to maximize Urgency(i) / (1 + Load(m))
        # This prioritizes jobs with lots of work, but penalizes using heavily loaded machines.
        
        best_idx = 0
        best_machine = None
        best_score = -1e10
        
        for i, op in enumerate(ready_operations):
            rw = op.get("remaining_work", 0)
            mo = op.get("machine_options", {})
            if not mo:
                continue
                
            # Find the machine that maximizes the score for this operation
            # Score = rw / (1 + load_penalty * load_m)
            # We can choose the machine, so we pick the best machine for this op.
            best_m_for_op = None
            best_m_score_for_op = -1e10
            
            for m, dur in mo.items():
                load_m = _machine_loads.get(m, 0)
                # Heuristic: Penalize load
                # Also, shorter duration is generally good for flow, but for makespan,
                # balancing load is key.
                # Let's use: Score = rw / (1 + 0.5 * load_m)
                # The 0.5 is a tuning constant for load sensitivity.
                score_m = rw / (1.0 + 0.5 * load_m)
                
                if score_m > best_m_score_for_op:
                    best_m_score_for_op = score_m
                    best_m_for_op = m
            
            if best_m_for_op is not None and best_m_score_for_op > best_score:
                best_score = best_m_score_for_op
                best_idx = i
                best_machine = best_m_for_op
        
        # If we found a best machine, update our internal load estimate
        if best_machine is not None:
            # We need to know the duration to update the load.
            # Find the duration we just picked.
            chosen_op = ready_operations[best_idx]
            chosen_dur = chosen_op["machine_options"][best_machine]
            _machine_loads[best_machine] = _machine_loads.get(best_machine, 0) + chosen_dur
            
            # Optional: Decay other loads? 
            # If we don't decay, loads will accumulate indefinitely.
            # We should reset or decay when time advances significantly.
            # But we don't know the time delta reliably if the simulator jumps.
            # Let's add a simple decay: if time has advanced, reduce loads slightly?
            # Actually, if a machine is in 'machine_options', it is free.
            # This implies its previous work is done.
            # So, if we see a machine in 'machine_options', we should arguably reset its load to 0?
            # No, because other jobs might have been assigned to it in the past and are still running?
            # No, if it's in 'machine_options', it is FREE.
            # So, any machine present in the current 'ready_operations' machine_options is currently idle.
            # Therefore, its load from previous assignments is no longer relevant for "current" congestion?
            # Wait, if it's idle, its load is 0.
            # So why track load?
            # Because we might pick a machine that is NOT in the current list? No, we can only pick from the list.
            # So, if we can only pick from available machines, and available machines are idle,
            # then "load" is always 0 at the moment of decision?
            
            # This logic suggests that tracking load of *idle* machines is useless.
            # The bottleneck is determined by the *sequence* of assignments.
            # If we assign Job A to M1, M1 becomes busy.
            # Later, when M1 is free again, it will appear in machine_options.
            # At that point, we don't know how long it was busy, but we know it's free now.
            
            # So, the "Load Balancing" heuristic must rely on something else.
            # Perhaps: **Minimize Total Weighted Flow Time**.
            # Or simply: **Shortest Processing Time (SPT)** for the selected job.
            # Why SPT? It frees machines faster, allowing more parallelism.
            
            # Let's pivot to a **SPT-based** structural change.
            # Select the operation with the shortest duration on its best machine?
            # No, that ignores job criticality.
            
            # Let's try: **Critical Ratio (CR)**
            # CR = (Time to Due Date) / (Remaining Work).
            # We don't have Due Dates.
            
            # Let's try: **Most Work Remaining (MWR)** with **SPT** tie-breaking.
            # The current program does MWR / min_dur.
            
            # Let's look at the "Greedy SPT-based load simulation" insight again.
            # "Greedy SPT-based load simulation that better predicts future machine congestion"
            # This implies we should simulate the effect of our choice.
            
            # Since we can't simulate the whole future, let's use a **Lookahead Heuristic**.
            # Pick the op that minimizes the maximum completion time of the next 2 steps?
            # Too complex.
            
            # Let's stick to the **Load Balancing** idea but fix the logic.
            # If we assume that picking a machine with a shorter duration reduces the "blocking time" for that machine,
            # we should prefer shorter durations for the same job.
            # And for different jobs, prefer those with higher remaining work.
            
            # The current program does: Score = rw / (min_dur + 1).
            # This penalizes long durations.
            
            # What if we use: Score = rw * min_dur? (Prioritize long ops for critical jobs?)
            # Or Score = rw / min_dur? (Current)
            
            # Let's try a **Weighted Shortest Processing Time (WSPT)** rule.
            # WSPT is optimal for single machine flow time.
            # For makespan in job shop, it's a good heuristic.
            # WSPT = Work / Duration.
            # Here, "Work" is remaining_work. "Duration" is the duration on the chosen machine.
            # We want to MAXIMIZE WSPT?
            # High WSPT means high work per unit time.
            
            # Let's implement WSPT:
            # For each op, find the machine that maximizes (rw / dur).
            # Then pick the op with the highest such value.
            
            best_idx = 0
            best_machine = None
            best_score = -1e10
            
            for i, op in enumerate(ready_operations):
                rw = op.get("remaining_work", 0)
                mo = op.get("machine_options", {})
                if not mo:
                    continue
                
                best_m_for_op = None
                best_m_score_for_op = -1e10
                
                for m, dur in mo.items():
                    if dur == 0:
                        continue
                    # WSPT score
                    score_m = rw / dur
                    
                    # Add a small bonus for machines that are "less used" in our internal tracking?
                    # Even if they are idle now, maybe they were busy a lot recently?
                    # Let's keep the load tracking as a tie-breaker or minor penalty.
                    load_m = _machine_loads.get(m, 0)
                    # Penalty for high historical load
                    adjusted_score = score_m / (1.0 + 0.1 * load_m)
                    
                    if adjusted_score > best_m_score_for_op:
                        best_m_score_for_op = adjusted_score
                        best_m_for_op = m
                
                if best_m_for_op is not None and best_m_score_for_op > best_score:
                    best_score = best_m_score_for_op
                    best_idx = i
                    best_machine = best_m_for_op
            
            if best_machine is not None:
                chosen_op = ready_operations[best_idx]
                chosen_dur = chosen_op["machine_options"][best_machine]
                _machine_loads[best_machine] = _machine_loads.get(best_machine, 0) + chosen_dur
                # Decay all loads slightly to prevent infinite growth
                for m in list(_machine_loads.keys()):
                    _machine_loads[m] *= 0.99
                return (best_idx, best_machine)
            
            # Fallback
            return best_idx

    except Exception:
        if not ready_operations:
            return 0
        return 0

# EVOLVE-BLOCK-END
