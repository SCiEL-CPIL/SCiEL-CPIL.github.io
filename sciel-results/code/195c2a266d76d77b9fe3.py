    def __call__(self, func):
        # ... setup ...
        
        # Main loop
        # ...
        
        # Polish
        if calls < budget:
            # Use scipy.optimize.minimize
            # We need to wrap func to count calls and stop.
            call_count = [calls]
            def count_func(x):
                if call_count[0] >= budget:
                    return func(x) # This will be ignored if we break, but scipy might not stop cleanly
                call_count[0] += 1
                return func(x)
            
            # This is risky.
    