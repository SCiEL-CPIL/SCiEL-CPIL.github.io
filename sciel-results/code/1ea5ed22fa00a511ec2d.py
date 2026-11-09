<<<<<<< SEARCH
        # Track fitness history for stagnation detection
        no_improve = 0
        stagnation_threshold = 50

        # Evaluate initial point
        x_init = np.zeros(d)
        fx_init = func(x_init)
        calls_made += 1
        best_x = x_init.copy()
        best_f = fx_init
        x = x_init.copy()
=======
        # Track fitness history for stagnation detection
        no_improve = 0
        stagnation_threshold = 15  # Earlier restart to explore more

        # Evaluate initial point
        x_init = np.zeros(d)
        fx_init = func(x_init)
        calls_made += 1
        best_x = x_init.copy()
        best_f = fx_init
        x = x_init.copy()
>>>>>>> REPLACE

