import numpy as np

def dispatch(ready_operations, time):
        if not ready_operations:
            return 0

        best_score = -1
        best_index = 0
        best_machine = None

        for i, op in enumerate(ready_operations):
            remaining_work = op.get("remaining_work", 0)
            machine_options = op.get("machine_options", {})
            
            if not machine_options:
                continue
            
            # Find the machine with the shortest duration for this operation
            min_duration = float('inf')
            best_m_for_op = None
            for m, d in machine_options.items():
                if d < min_duration:
                    min_duration = d
                    best_m_for_op = m
            
            if min_duration == float('inf'):
                continue
                
            # Heuristic: Prioritize high remaining work, penalize long processing times
            # Ratio: Work / Time. Higher is better.
            score = remaining_work / (min_duration + 1e-9)
            
            if score > best_score:
                best_score = score
                best_index = i
                best_machine = best_m_for_op
        
        # If no valid operation found, fallback to first
        if best_machine is None:
            return 0
        
        return (best_index, best_machine)
