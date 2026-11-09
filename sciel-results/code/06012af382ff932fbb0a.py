
One optimization: `random.seed` inside the loop might be slow or deterministic in a bad way if `restart_count` is the only variable. Using `random.seed(None)` or a specific salt is better.
Also, the `all_pairs` generation and shuffle takes time. We can pre-generate the list of pairs.

Let's refine the loop to use a pre-generated list of pairs and just shuffle it.

