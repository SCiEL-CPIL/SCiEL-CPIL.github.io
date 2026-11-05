    def get_attr(dag, name, default=None):
        if isinstance(dag, dict):
            return dag.get(name, default)
        else:
            return getattr(dag, name, default)
    